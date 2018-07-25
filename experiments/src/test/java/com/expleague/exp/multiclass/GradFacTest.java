package com.expleague.exp.multiclass;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.Pair;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.GridTools;
import com.expleague.ml.cli.output.printers.MulticlassProgressPrinter;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.factorization.impl.ALS;
import com.expleague.ml.factorization.impl.ElasticNetFactorization;
import com.expleague.ml.factorization.impl.SVDAdapterEjml;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.loss.LogL2;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.meta.impl.fake.FakeTargetMeta;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.MultiClass;
import com.expleague.ml.methods.multiclass.gradfac.FMCBoosting;
import com.expleague.ml.methods.multiclass.gradfac.GradFacMulticlass;
import com.expleague.ml.methods.multiclass.gradfac.GradFacSvdNMulticlass;
import com.expleague.ml.methods.multiclass.gradfac.MultiClassColumnBootstrapOptimization;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.MultiClassModel;
import com.expleague.ml.testUtils.TestResourceLoader;
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
      pool.addTarget(TargetMeta.create("letter", "", FeatureMeta.ValueType.INTS),
          VecTools.toIntSeq(pool.target(L2.class).target)
      );
      final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100500), 0.9, 0.1);
      learn = pool.sub(idxs[0]);
      test = pool.sub(idxs[1]);
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
            new ALS(20),
            LogL2.class
        ),
        L2.class,
        10000,
        14
    );
    fitModel(boosting);
  }

  public void testGradFacSALS() throws Exception {
    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(
        new GradFacMulticlass(
            new GreedyObliviousTree<>(GridTools.medianGrid(learn.vecData(), 32), 5),
            new StochasticALS(new FastRandom(0), 100),
            LogL2.class
        ),
        L2.class,
        10000,
        5
    );
    fitModel(boosting);
  }

  public void testOVR() throws Exception {
    final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(
            new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
            2000,
            5
    );
    final VecDataSet vecDataSet = learn.vecData();
    final LLLogit globalLoss = learn.target(LLLogit.class);
//    final MulticlassProgressPrinter multiclassProgressPrinter = new MulticlassProgressPrinter(learn, test);
//    boosting.addListener(multiclassProgressPrinter);

    Interval.start();
    final Ensemble ensemble = boosting.fit(vecDataSet, globalLoss);
    Interval.stopAndPrint();
//
//    Interval.start();
//    System.out.println(MCTools.evalModel(multiclassModel, learn, "[LEARN] ", false));
//    System.out.println(MCTools.evalModel(multiclassModel, test, "[TEST] ", false));
//    Interval.stopAndPrint();
//    System.out.println(multiclassModel.count + " times");
  }

  public void testFMCBoostSALS() throws Exception {
    final FMCBoosting boosting = new FMCBoosting(
            new StochasticALS(new FastRandom(0), 100),
            new GreedyObliviousTree<L2>(GridTools.medianGrid(learn.vecData(), 32), 5),
            L2.class,
            2500,
            5
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
        20000,
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
        2000,
        7
    );
    fitModel(boosting);
  }

  private void fitModel(final GradientBoosting<BlockwiseMLLLogit> boosting) {
    final VecDataSet vecDataSet = learn.vecData();
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final MulticlassProgressPrinter multiclassProgressPrinter = new MulticlassProgressPrinter(learn, test);
//    boosting.addListener(multiclassProgressPrinter);

    Interval.start();
    final Ensemble ensemble = boosting.fit(vecDataSet, globalLoss);
    Interval.stopAndPrint();

    final MultiClassModel multiclassModel;
    if (ensemble.last() instanceof FuncJoin) {
      final FuncJoin joined = MCTools.joinBoostingResult(ensemble);
      multiclassModel = new MultiClassModel(joined);
    }
    else
      multiclassModel = new MultiClassModel(ensemble);

    Interval.start();
    System.out.println(MCTools.evalModel(multiclassModel, learn, "[LEARN] ", false));
    System.out.println(MCTools.evalModel(multiclassModel, test, "[TEST] ", false));
    Interval.stopAndPrint();
    System.out.println(multiclassModel + " times");
  }

  private void fitModel(final FMCBoosting boosting) {
    final VecDataSet vecDataSet = learn.vecData();
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final MulticlassProgressPrinter multiclassProgressPrinter = new MulticlassProgressPrinter(learn, test);
    boosting.addListener(multiclassProgressPrinter);

    final Ensemble ensemble = boosting.fit(vecDataSet, globalLoss);
    final Trans joined = ensemble.last() instanceof FuncJoin ? MCTools.joinBoostingResult(ensemble) : ensemble;
    final MultiClassModel multiclassModel = new MultiClassModel(joined);
    final String learnResult = MCTools.evalModel(multiclassModel, learn, "[LEARN] ", false);
    final String testResult = MCTools.evalModel(multiclassModel, test, "[TEST] ", false);
    System.out.println(learnResult);
    System.out.println(testResult);
  }

  private static void applyFactorMethod(final Mx x, final Factorization method) {
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
