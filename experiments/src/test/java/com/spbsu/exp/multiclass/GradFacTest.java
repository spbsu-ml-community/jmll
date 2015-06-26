package com.spbsu.exp.multiclass;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.cli.output.printers.MulticlassProgressPrinter;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.SubPool;
import com.spbsu.ml.factorization.OuterFactorization;
import com.spbsu.ml.factorization.impl.ALS;
import com.spbsu.ml.factorization.impl.ElasticNetFactorization;
import com.spbsu.ml.factorization.impl.SVDAdapterEjml;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LogL2;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.fake.FakeTargetMeta;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.MultiClass;
import com.spbsu.ml.methods.multiclass.gradfac.GradFacMulticlass;
import com.spbsu.ml.methods.multiclass.gradfac.GradFacSvdNMulticlass;
import com.spbsu.ml.methods.multiclass.gradfac.MultiClassColumnBootstrapOptimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.testUtils.TestResourceLoader;
import junit.framework.TestCase;

import java.io.IOException;

/**
 * User: qdeee
 * Date: 27.02.15
 */
public class GradFacTest extends TestCase {
  private static Pool<?> learn;
  private static Pool<?> test;

  private synchronized static void init() throws IOException {
    if (learn == null || test == null) {
      final Pool<?> pool = TestResourceLoader.loadPool("multiclass/ds_letter/letter.tsv.gz");
      pool.addTarget(new FakeTargetMeta(pool.vecData(), FeatureMeta.ValueType.INTS),
          VecTools.toIntSeq(pool.target(L2.class).target)
      );
      final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100500), 0.8, 0.5);
      learn = new SubPool<>(pool, idxs[0]);
      test = new SubPool<>(pool, idxs[1]);
    }
  }

  @Override
  protected void setUp() throws Exception {
    init();
  }


  public void testGradMxApproxSVDN() throws Exception {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = (Mx) globalLoss.gradient(new ArrayVec(globalLoss.dim()));
    double time = System.currentTimeMillis();
    int factorDim = 1;
//    for (int factorDim = gradient.columns(); factorDim >= 1; factorDim--)
    {
      final Pair<Vec, Vec> pair = new SVDAdapterEjml(factorDim).factorize(gradient);
      final Mx h = (Mx) pair.getFirst();
      final Mx b = (Mx) pair.getSecond();
      System.out.println("factor dim: " + factorDim);
      System.out.println("time: " + ((System.currentTimeMillis() - time) / 1000));
      final Mx afterFactor = MxTools.multiply(h, MxTools.transpose(b));
      System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(gradient, afterFactor) + ", l1 = " + VecTools.distanceL1(gradient, afterFactor));
      System.out.println();
    }
  }
  public void testElasticNetGradFac() throws Exception {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = (Mx) globalLoss.gradient(new ArrayVec(globalLoss.dim()));
    final ElasticNetFactorization elasticNetFactorization = new ElasticNetFactorization(20, 1e-2, 0.95, 0.15 * 1e-6);
    final Pair<Vec, Vec> pair = elasticNetFactorization.factorize(gradient);
    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();
    final Mx afterFactor = VecTools.outer(h, b);
    System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(gradient, afterFactor) + ", l1 = " + VecTools.distanceL1(gradient, afterFactor));
    System.out.println();
  }

  private static class ParameterCollector {
    double lambda;
    double alpha;
    double l2;
    double l1;

    public ParameterCollector(final double lambda, final double alpha, final double l2, final double l1) {
      this.lambda = lambda;
      this.alpha = alpha;
      this.l2 = l2;
      this.l1 = l1;
    }

    @Override
    public String toString() {
      return "ParameterCollector{" +
          "lambda=" + lambda +
          ", alpha=" + alpha +
          ", l2=" + l2 +
          ", l1=" + l1 +
          '}';
    }
  }

  public void testElasticNetGradFacGridSearch() throws Exception {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = (Mx) globalLoss.gradient(new ArrayVec(globalLoss.dim()));

    ParameterCollector minL1ParameterCollector = new ParameterCollector(0, 0, Double.MAX_VALUE, Double.MAX_VALUE);
    ParameterCollector minL2ParameterCollector = new ParameterCollector(0, 0, Double.MAX_VALUE, Double.MAX_VALUE);

    for (double lambda = 0.15 * 1e-7; lambda < 1e-4; lambda += 1e-6) {
      for (double alpha = 0.1; alpha < 1.0; alpha += 0.01) {
        final ElasticNetFactorization elasticNetFactorization = new ElasticNetFactorization(20, 1e-2, 0.95, 0.15 * 1e-6);
        final Pair<Vec, Vec> pair = elasticNetFactorization.factorize(gradient);
        final Vec h = pair.getFirst();
        final Vec b = pair.getSecond();
        final Mx afterFactor = VecTools.outer(h, b);
        final double l2 = VecTools.distance(gradient, afterFactor);
        final double l1 = VecTools.distanceL1(gradient, afterFactor);

        if (l2 < minL2ParameterCollector.l2) {
          minL2ParameterCollector = new ParameterCollector(lambda, alpha, l2, l1);
        }
        if (l1 < minL1ParameterCollector.l1) {
          minL1ParameterCollector = new ParameterCollector(lambda, alpha, l2, l1);
        }

      }
    }
    System.out.println(minL2ParameterCollector.toString());
    System.out.println(minL1ParameterCollector.toString());
  }

  public void testSimpleMx() throws Exception {
    final Mx mx = genUniformRandMx(5, 3, 100500);

    final ElasticNetFactorization elasticNetFactorization = new ElasticNetFactorization(1, 1e-1, 0.0, 0.0);
    final Pair<Vec, Vec> pair = elasticNetFactorization.factorize(mx);
    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();
    final Mx afterFactor = VecTools.outer(h, b);
    System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(mx, afterFactor) + ", l1 = " + VecTools.distanceL1(mx, afterFactor));
  }

  private static Mx genUniformRandMx(final int m, final int n, final int seed) {
    final Mx mx = new VecBasedMx(m, n);
    final FastRandom fastRandom = new FastRandom(seed);
    for (int i = 0; i < mx.dim(); i++) {
      mx.set(i, fastRandom.nextDouble());
    }
    return mx;
  }

  public void testDifferentMethods() throws Exception {
    final Mx mx = genUniformRandMx(500, 300, 100500);

    applyFactorMethod(mx, new ALS(15));
    applyFactorMethod(mx, new SVDAdapterEjml());
    final double lambda = 0.0015;
    applyFactorMethod(mx, new ElasticNetFactorization(20, 1e-4, 0.5, lambda));
  }

  public void testGradFacBaseline() throws Exception {
    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(
        new GradFacMulticlass(
            new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
            new SVDAdapterEjml(1),
            LogL2.class
        ),
        L2.class,
        400,
        7
    );
    fitModel(boosting);
  }

  public void testGradFacElasticNet() throws Exception {
    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(
        new GradFacMulticlass(
            new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
            new ElasticNetFactorization(1, 1., 1., 1.),
            SatL2.class
        ),
        L2.class,
        400,
        7
    );
    fitModel(boosting);
  }

  public void testGradFacSVDNColumnsBootstrap() throws Exception {
    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(
        new MultiClassColumnBootstrapOptimization(
            new GradFacSvdNMulticlass(
                new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
                LogL2.class,
                2
            ),
            new FastRandom(100500),
            1.
        ),
        L2.class,
        5000,
        5
    );
    fitModel(boosting);
  }

  public void testGradFacColumnsBootstrap() throws Exception {
    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(
        new MultiClassColumnBootstrapOptimization(
            new GradFacMulticlass(
                new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
                new SVDAdapterEjml(1),
                SatL2.class
            ), new FastRandom(),
            1.
        ),
        L2.class,
        7500,
        7
    );
    fitModel(boosting);
  }

  public void testGradFacElasticNetColumnsBootstrap() throws Exception {
    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(
        new MultiClassColumnBootstrapOptimization(
            new GradFacMulticlass(
                new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
                new ElasticNetFactorization(20, 1e-2, 0.95, 0.15 * 1e-6),
                LogL2.class,
                true
            ),
            new FastRandom(100500),
            1.
        ),
        L2.class,
        5000,
        7
    );
    fitModel(boosting);
  }

  public void testBaseline() throws Exception {
    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(
        new MultiClass(
            new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
            LogL2.class
        ),
        L2.class,
        400,
        0.3
    );
    fitModel(boosting);
  }

  private void fitModel(final GradientBoosting<BlockwiseMLLLogit> boosting) {
    final VecDataSet vecDataSet = learn.vecData();
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final MulticlassProgressPrinter multiclassProgressPrinter = new MulticlassProgressPrinter(learn, test);
    boosting.addListener(multiclassProgressPrinter);

    final Ensemble ensemble = boosting.fit(vecDataSet, globalLoss);

    final FuncJoin joined = MCTools.joinBoostingResult(ensemble);
    final MultiClassModel multiclassModel = new MultiClassModel(joined);
    final String learnResult = MCTools.evalModel(multiclassModel, learn, "[LEARN] ", false);
    final String testResult = MCTools.evalModel(multiclassModel, test, "[TEST] ", false);
    System.out.println(learnResult);
    System.out.println(testResult);
  }

  private static void applyFactorMethod(final Mx x, final OuterFactorization method) {
    final Pair<Vec, Vec> pair = method.factorize(x);
    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();
    final double normB = VecTools.norm(b);
    VecTools.scale(b, 1 / normB);
    VecTools.scale(h, normB);
    final Mx afterFactor = VecTools.outer(h, b);
    System.out.println(method.getClass().getSimpleName() + ": ||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(x, afterFactor) + ", l1 = " + VecTools.distanceL1(x, afterFactor));
  }
}
