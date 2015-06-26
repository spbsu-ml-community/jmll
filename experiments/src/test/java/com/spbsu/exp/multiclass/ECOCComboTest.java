package com.spbsu.exp.multiclass;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.Pair;
import com.spbsu.exp.multiclass.spoc.full.mx.optimization.ECOCMulticlass;
import com.spbsu.exp.multiclass.spoc.full.mx.optimization.SeparatedMLLLogit;
import com.spbsu.exp.multiclass.weak.CustomWeakBinClass;
import com.spbsu.exp.multiclass.weak.CustomWeakMultiClass;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.cli.output.printers.MulticlassProgressPrinter;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.SubPool;
import com.spbsu.ml.factorization.OuterFactorization;
import com.spbsu.ml.factorization.impl.ALS;
import com.spbsu.ml.factorization.impl.SVDAdapterEjml;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.fake.FakeTargetMeta;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.multiclass.gradfac.GradFacMulticlass;
import com.spbsu.ml.methods.multiclass.gradfac.GradFacSvdNMulticlass;
import com.spbsu.ml.methods.multiclass.spoc.ECOCCombo;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.models.multiclass.MCModel;
import com.spbsu.ml.models.multiclass.MulticlassCodingMatrixModel;
import com.spbsu.ml.testUtils.TestResourceLoader;
import junit.framework.TestCase;

import java.io.FileInputStream;
import java.io.IOException;

public class ECOCComboTest extends TestCase {
  private static Pool<?> learn;
  private static Pool<?> test;
  private static Mx S;

  private synchronized static void init() throws IOException {
    if (learn == null || test == null) {
      final Pool<?> pool = TestResourceLoader.loadPool("multiclass/ds_letter/letter.tsv.gz");
      pool.addTarget(new FakeTargetMeta(pool.vecData(), FeatureMeta.ValueType.INTS),
                     VecTools.toIntSeq(pool.target(L2.class).target)
      );
      final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100500), 0.8, 0.5);
      learn = new SubPool<>(pool, idxs[0]);
      test = new SubPool<>(pool, idxs[1]);

      final CharSequence mxStr = StreamTools.readStream(new FileInputStream("/Users/qdeee/datasets/catalog-final/catalog50-1stlevel-gt5000.tsv.simmx"));
      S = MathTools.CONVERSION.convert(mxStr, Mx.class);

    }
  }

  @Override
  protected void setUp() throws Exception {
    init();
  }

  public void testFit() throws Exception {
    final BlockwiseMLLLogit mllLogit = learn.target(BlockwiseMLLLogit.class);
    final VecDataSet vecDataSet = learn.vecData();

    final int k = MCTools.countClasses(mllLogit.labels());
    final ECOCCombo ecocComboMethod = new ECOCCombo(k, k, 5.0, 2.5, 3.0, S, new CustomWeakBinClass(100, 0.3));
    final Action<MulticlassCodingMatrixModel> listener = new Action<MulticlassCodingMatrixModel>() {
      @Override
      public void invoke(final MulticlassCodingMatrixModel model) {
        System.out.println("L == " + model.getInternalModel().ydim());
        System.out.println(getPairwiseInteractions(model));
        System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", true));
        System.out.println(MCTools.evalModel(model, test, "[TEST] ", true));
      }
    };
    ecocComboMethod.addListener(listener);
    final MulticlassCodingMatrixModel model = (MulticlassCodingMatrixModel) ecocComboMethod.fit(vecDataSet, mllLogit);
    System.out.println("\n\n\n");
    System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST] ", false));

  }

  public void testDefaultGreedyModelFit() throws Exception {
    final BlockwiseMLLLogit mllLogit = learn.target(BlockwiseMLLLogit.class);
    final VecDataSet vecDataSet = learn.vecData();

    final int k = MCTools.countClasses(mllLogit.labels());
    final ECOCCombo ecocComboMethod = new ECOCCombo(k, k, 5.0, 2.5, 3.0, S, new CustomWeakBinClass(100, 0.3));
    final Action<MulticlassCodingMatrixModel> listener = new Action<MulticlassCodingMatrixModel>() {
      @Override
      public void invoke(final MulticlassCodingMatrixModel model) {
        System.out.println("L == " + model.getInternalModel().ydim());
        System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", true));
        System.out.println(MCTools.evalModel(model, test, "[TEST] ", true));
        System.out.println();
      }
    };
    ecocComboMethod.addListener(listener);
    final MulticlassCodingMatrixModel model = (MulticlassCodingMatrixModel) ecocComboMethod.fit(vecDataSet, mllLogit);
    System.out.println("\n\n\n");
    System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST] ", false));

  }

  public void testFitUpdatePrior() throws Exception {
    final double lambda = 0.9;
    final BlockwiseMLLLogit mllLogit = learn.target(BlockwiseMLLLogit.class);
    final VecDataSet vecDataSet = learn.vecData();

    final int k = MCTools.countClasses(mllLogit.labels());
    final ECOCCombo ecocComboMethod = new ECOCCombo(k, k, 5.0, 2.5, 3.0, S, new CustomWeakBinClass(100, 0.3));
    final Action<MulticlassCodingMatrixModel> listener = new Action<MulticlassCodingMatrixModel>() {
      @Override
      public void invoke(final MulticlassCodingMatrixModel model) {
        final Mx mx = getPairwiseInteractions(model);
        VecTools.scale(S, lambda);
        VecTools.scale(mx, 1 - lambda);
        VecTools.append(S, mx);

        System.out.println("L == " + model.getInternalModel().ydim());
        System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", true));
        System.out.println(MCTools.evalModel(model, test, "[TEST] ", true));
        System.out.println();
      }
    };
    ecocComboMethod.addListener(listener);
    final MulticlassCodingMatrixModel model = (MulticlassCodingMatrixModel) ecocComboMethod.fit(vecDataSet, mllLogit);
    System.out.println("\n\n\n");
    System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST] ", false));

  }

  private static Mx getPairwiseInteractions(final MCModel model) {
    final Mx result = new VecBasedMx(S.columns(), S.rows());

    final Mx features = learn.vecData().data();
    final int[] counts = new int[features.rows()];
    for (int i = 0; i < learn.size(); i++) {
      final Vec probs = model.probs(features.row(i));
      final int bestClass = VecTools.argmax(probs);
      VecTools.append(result.row(bestClass), probs);
      counts[bestClass]++;
    }
    for (int c = 0; c < result.rows(); c++) {
      VecTools.scale(result.row(c), 1.0 / counts[c]);
    }
    for (int c1 = 0; c1 < result.rows(); c1++) {
      for (int c2 = c1 + 1; c2 < result.columns(); c2++) {
        final double val = 0.5 * (result.get(c1, c2) + result.get(c2, c1));
        result.set(c1, c2, val);
        result.set(c2, c1, val);
      }
    }
    return result;
  }

  public void testBoostedECOC() throws Exception {
    final VecDataSet vecDataSet = learn.vecData();
    final IntSeq labels = learn.target(BlockwiseMLLLogit.class).labels();

    final BFGrid bfGrid = GridTools.medianGrid(vecDataSet, 32);
    final SeparatedMLLLogit smlllogit = new SeparatedMLLLogit(5, labels, null);

    final int k = MCTools.countClasses(smlllogit.labels());
    final int l = smlllogit.getBinClassifiersCount();

    final ECOCMulticlass ecocMulticlass = new ECOCMulticlass(new GreedyObliviousTree<L2>(bfGrid, 5), SatL2.class, k, l, 1.0);
    final GradientBoosting<SeparatedMLLLogit> boosting = new GradientBoosting<>(ecocMulticlass, 3, 0.1);
    final Ensemble fit = boosting.fit(vecDataSet, smlllogit);
    System.out.println(fit);
  }

  public void testMCMMProbs() throws Exception {
    final BlockwiseMLLLogit mllLogit = learn.target(BlockwiseMLLLogit.class);
    final VecDataSet vecDataSet = learn.vecData();

    final int k = MCTools.countClasses(mllLogit.labels());
    final ECOCCombo ecocComboMethod = new ECOCCombo(k, 5, 5.0, 2.5, 3.0, S, new CustomWeakBinClass(10, 0.3));
    final MulticlassCodingMatrixModel model = (MulticlassCodingMatrixModel) ecocComboMethod.fit(vecDataSet, mllLogit);

    for (int i = 0; i < vecDataSet.data().rows(); i++) {
      final Vec features = vecDataSet.data().row(i);
      final Vec probs = model.probs(features);
      final int bestClass = model.bestClass(features);
      assertEquals(bestClass, VecTools.argmax(probs));
    }
  }

  public void testGradFacALS() throws Exception {
    final VecDataSet vecDataSet = learn.vecData();
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final BFGrid bfGrid = GridTools.medianGrid(vecDataSet, 32);
    final int factorIters = 15;
    final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new GradFacMulticlass(new GreedyObliviousTree<L2>(bfGrid, 5),
        new ALS(factorIters), SatL2.class), 500, 0.7);
    boosting.addListener(new MulticlassProgressPrinter(learn, test));

    final Ensemble ensemble = boosting.fit(vecDataSet, globalLoss);

    final FuncJoin joined = MCTools.joinBoostingResult(ensemble);
    final MultiClassModel multiclassModel = new MultiClassModel(joined);
    final String learnResult = MCTools.evalModel(multiclassModel, learn, "[LEARN] ", false);
    final String testResult = MCTools.evalModel(multiclassModel, test, "[TEST] ", false);
    System.out.println(learnResult);
    System.out.println(testResult);

  }

  public void testGradFacSvdN() throws Exception {
    final VecDataSet vecDataSet = learn.vecData();
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final BFGrid bfGrid = GridTools.medianGrid(vecDataSet, 32);
    final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new GradFacSvdNMulticlass(new GreedyObliviousTree<L2>(bfGrid, 5),
        SatL2.class, 2), 500, 0.7);
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

  public void testBaseline() throws Exception {
    final VecDataSet vecDataSet = learn.vecData();
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(300, 0.3);
    final MCModel model = (MCModel) customWeakMultiClass.fit(vecDataSet, globalLoss);
    System.out.println(MCTools.evalModel(model, learn, "[LEARN]", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST]", false));
  }

  public void testGradMxApproxALS() throws Exception {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = (Mx) globalLoss.gradient(new ArrayVec(globalLoss.dim()));

    final int factorIters = 25;
    final ALS als = new ALS(factorIters);
    final Action<Pair<Vec, Vec>> action = new Action<Pair<Vec, Vec>>() {
      @Override
      public void invoke(final Pair<Vec, Vec> pair) {
        final Vec h = pair.getFirst();
        final Vec b = pair.getSecond();
        System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", RMSE = " + rmse(gradient, VecTools.outer(h, b)));
      }
    };
    als.addListener(action);
    als.factorize(gradient);
  }

  public void testGradMxApproxSVD() throws Exception {
    applyFactorMethod(new ALS(15));
    applyFactorMethod(new SVDAdapterEjml());
  }

  public void testGradMxApproxSVDN() throws Exception {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = (Mx) globalLoss.gradient(new ArrayVec(globalLoss.dim()));
    double time = System.currentTimeMillis();

    for (int factorDim = gradient.columns(); factorDim >= 1; factorDim--)
    {
      final Pair<Vec, Vec> pair = new SVDAdapterEjml(factorDim).factorize(gradient);
      final Mx h = (Mx) pair.getFirst();
      final Mx b = (Mx) pair.getSecond();
      System.out.println("factor dim: " + factorDim);
      System.out.println("time: " + ((System.currentTimeMillis() - time) / 1000));
      final Mx afterFactor = MxTools.multiply(h, MxTools.transpose(b));
//      System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(gradient, afterFactor) + ", l1 = " + VecTools.distanceL1(gradient, afterFactor));
      System.out.println();
    }
  }

  private static void applyFactorMethod(final OuterFactorization method) {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = (Mx) globalLoss.gradient(new ArrayVec(globalLoss.dim()));
    final Pair<Vec, Vec> pair = method.factorize(gradient);
    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();
    final double normB = VecTools.norm(b);
    VecTools.scale(b, 1 / normB);
    VecTools.scale(h, normB);
    final Mx afterFactor = VecTools.outer(h, b);
//    System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", l2 = " + VecTools.distance(gradient, afterFactor) + ", l1 = " + VecTools.distanceL1(gradient, afterFactor));
  }

  private static double rmse(final Vec target, final Vec approx) {
    return Math.sqrt(VecTools.sum2(VecTools.subtract(target, approx)) / target.length());
  }


}