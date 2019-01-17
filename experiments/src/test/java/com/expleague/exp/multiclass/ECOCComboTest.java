package com.expleague.exp.multiclass;

import com.expleague.exp.multiclass.spoc.full.mx.optimization.ECOCMulticlass;
import com.expleague.commons.io.StreamTools;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.util.Pair;
import com.expleague.commons.util.logging.Interval;
import com.expleague.exp.multiclass.spoc.full.mx.optimization.SeparatedMLLLogit;
import com.expleague.exp.multiclass.weak.CustomWeakBinClass;
import com.expleague.exp.multiclass.weak.CustomWeakMultiClass;
import com.expleague.ml.GridTools;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.cli.output.printers.MulticlassProgressPrinter;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.factorization.impl.ALS;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.BFGrid;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.multiclass.gradfac.GradFacMulticlass;
import com.expleague.ml.methods.multiclass.spoc.ECOCCombo;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.MultiClassModel;
import com.expleague.ml.models.multiclass.MCModel;
import com.expleague.ml.models.multiclass.MulticlassCodingMatrixModel;
import com.expleague.ml.testUtils.TestResourceLoader;
import junit.framework.TestCase;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.function.Consumer;

public class ECOCComboTest extends TestCase {
  private static Pool<?> learn;
  private static Pool<?> test;
  private static Mx S;

  private synchronized static void init() throws IOException {
    if (learn == null || test == null) {
      final Pool<?> pool = TestResourceLoader.loadPool("multiclass/ds_letter/letter.tsv.gz");
      pool.addTarget(TargetMeta.create("letter", "", FeatureMeta.ValueType.INTS),
                     VecTools.toIntSeq(pool.target(L2.class).target)
      );
      final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100500), 0.8, 0.5);
      learn = pool.sub(idxs[0]);
      test = pool.sub(idxs[1]);

      final CharSequence mxStr = StreamTools.readStream(new FileInputStream("/Users/qdeee/datasets/catalog-final/catalog50-1stlevel-gt5000.tsv.simmx"));
      S = MathTools.CONVERSION.convert(mxStr, Mx.class);

    }
  }

  @Override
  protected void setUp() throws Exception {
//    init();
  }

  public void testFit() throws Exception {
    final BlockwiseMLLLogit mllLogit = learn.target(BlockwiseMLLLogit.class);
    final VecDataSet vecDataSet = learn.vecData();

    final int k = MCTools.countClasses(mllLogit.labels());
    final ECOCCombo ecocComboMethod = new ECOCCombo(k, k, 5.0, 2.5, 3.0, S, new CustomWeakBinClass(100, 0.3));
    final Consumer<com.expleague.ml.models.multiclass.MulticlassCodingMatrixModel> listener = model -> {
      System.out.println("L == " + model.getInternalModel().ydim());
      System.out.println(getPairwiseInteractions(model));
      System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", true));
      System.out.println(MCTools.evalModel(model, test, "[TEST] ", true));
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
    final Consumer<com.expleague.ml.models.multiclass.MulticlassCodingMatrixModel> listener = model -> {
      System.out.println("L == " + model.getInternalModel().ydim());
      System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", true));
      System.out.println(MCTools.evalModel(model, test, "[TEST] ", true));
      System.out.println();
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
    final Consumer<com.expleague.ml.models.multiclass.MulticlassCodingMatrixModel> listener = model -> {
      final Mx mx = getPairwiseInteractions(model);
      VecTools.scale(S, lambda);
      VecTools.scale(mx, 1 - lambda);
      VecTools.append(S, mx);

      System.out.println("L == " + model.getInternalModel().ydim());
      System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", true));
      System.out.println(MCTools.evalModel(model, test, "[TEST] ", true));
      System.out.println();
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
    final Consumer<Pair<Vec,Vec>> action = pair -> {
      final Vec h = pair.getFirst();
      final Vec b = pair.getSecond();
      System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", RMSE = " + rmse(gradient, VecTools.outer(h, b)));
    };
    als.addListener(action);
    als.factorize(gradient);
  }

  public void testGradMxApproxSVD() throws Exception {
    applyFactorMethod(new ALS(15));
  }

  public void testStochasticALS() {
    final FastRandom rng = new FastRandom(0);
    final Mx X = new VecBasedMx(100000, 100);
    VecTools.fillGaussian(X, rng);
    final StochasticALS sals = new StochasticALS(rng, 1000);
    final ALS als = new ALS(10);
    Interval.start();
    final Pair<Vec, Vec> pair = sals.factorize(X);
    Interval.stopAndPrint("SALS");
    Interval.start();
    final Pair<Vec, Vec> reference = als.factorize(X);
    Interval.stopAndPrint("ALS");
    final VecBasedMx u = new VecBasedMx(pair.first.dim(), pair.first);
    final VecBasedMx v = new VecBasedMx(1, pair.second);
    Mx mx = VecTools.outer(u, v);

    final VecBasedMx u1 = new VecBasedMx(reference.first.dim(), reference.first);
    final VecBasedMx v1 = new VecBasedMx(1, reference.second);
    Mx mx1 = VecTools.outer(u1, v1);

    System.out.println(VecTools.norm(X));
    System.out.println(VecTools.distance(mx, X));
    System.out.println(VecTools.distance(mx1, X));
    assertTrue(VecTools.distance(mx, X) < VecTools.distance(mx1, X) + MathTools.EPSILON);
  }

  public void testStochasticALSElastic() {
    final FastRandom rng = new FastRandom(0);
    final Mx X = new VecBasedMx(100000, 100);
    VecTools.fillGaussian(X, rng);
    final StochasticALS sals = new StochasticALS(rng, 1000, 100000, 0.2, 0.1, null);
    final ALS als = new ALS(10);
    Interval.start();
    final Pair<Vec, Vec> pair = sals.factorize(X);
    Interval.stopAndPrint("SALS");
    Interval.start();
    final Pair<Vec, Vec> reference = als.factorize(X);
    Interval.stopAndPrint("ALS");
    final VecBasedMx u = new VecBasedMx(pair.first.dim(), pair.first);
    final VecBasedMx v = new VecBasedMx(1, pair.second);
    Mx mx = VecTools.outer(u, v);

    final VecBasedMx u1 = new VecBasedMx(reference.first.dim(), reference.first);
    final VecBasedMx v1 = new VecBasedMx(1, reference.second);
    Mx mx1 = VecTools.outer(u1, v1);

    System.out.println(VecTools.norm(X));
    System.out.println(VecTools.distance(mx, X));
    System.out.println(VecTools.distance(mx1, X));
    assertTrue(VecTools.distance(mx, X) < VecTools.distance(mx1, X) + MathTools.EPSILON);
  }

  private static void applyFactorMethod(final Factorization method) {
    final BlockwiseMLLLogit globalLoss = learn.target(BlockwiseMLLLogit.class);
    final Mx gradient = globalLoss.gradient(new ArrayVec(globalLoss.dim()));
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