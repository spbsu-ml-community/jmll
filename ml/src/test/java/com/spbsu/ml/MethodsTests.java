package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.util.logging.Interval;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.FeaturesTxtPool;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.func.NormalizedLinear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.L2GreedyTDRegion;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.greedyRegion.GreedyRegion;
import com.spbsu.ml.methods.greedyRegion.RegionForest;
import com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization.GreedyMergedRegion;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.pgm.ProbabilisticGraphicalModel;
import com.spbsu.ml.models.pgm.SimplePGM;
import gnu.trove.map.hash.TDoubleDoubleHashMap;
import gnu.trove.map.hash.TDoubleIntHashMap;

import java.util.Random;

import static com.spbsu.commons.math.MathTools.sqr;
import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: solar
 * Date: 26.11.12
 *
 * Time: 15:50
 */
public abstract class MethodsTests extends GridTest {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public void testPGMFit3x3() {
    final SimplePGM original = new SimplePGM(new VecBasedMx(3, new ArrayVec(new double[]{
            0, 0.2, 0.8,
            0, 0, 1.,
            0, 0, 0
    })));
    checkRestoreFixedTopology(original, PGMEM.GAMMA_PRIOR_PATH, 0.0, 100, 0.01);
  }

  public void testPGMFit5x5() {
    final SimplePGM original = new SimplePGM(new VecBasedMx(5, new ArrayVec(new double[]{
            0, 0.2, 0.3, 0.1, 0.4,
            0, 0, 0.25, 0.25, 0.5,
            0, 0, 0, 0.1, 0.9,
            0, 0, 0.5, 0, 0.5,
            0, 0, 0, 0, 0
    })));


    checkRestoreFixedTopology(original, PGMEM.GAMMA_PRIOR_PATH, 0., 100, 0.01);
  }

  public void testPGMFit5x5RandSkip() {
    final SimplePGM original = new SimplePGM(new VecBasedMx(5, new ArrayVec(new double[]{
            0, 0.2, 0.3, 0.1, 0.4,
            0, 0, 0.25, 0.25, 0.5,
            0, 0, 0, 0.1, 0.9,
            0, 0, 0.5, 0, 0.5,
            0, 0, 0, 0, 0
    })));

    checkRestoreFixedTopology(original, PGMEM.GAMMA_PRIOR_PATH, 0.8, 100, 0.01);
  }

  public void testPGMFit10x10Rand() {
    final VecBasedMx originalMx = new VecBasedMx(10, new ArrayVec(100));
    for (int i = 0; i < originalMx.rows() - 1; i++) {
      for (int j = 0; j < originalMx.columns(); j++)
        originalMx.set(i, j, rng.nextDouble() < 0.5 && j > 0 ? 1 : 0);
      VecTools.normalizeL1(originalMx.row(i));
    }
    VecTools.fill(originalMx.row(originalMx.rows() - 1), 0);
    final SimplePGM original = new SimplePGM(originalMx);
    checkRestoreFixedTopology(original, PGMEM.POISSON_PRIOR_PATH, 0.5, 100, 0.01);
  }

  public void testPGMFit10x10FreqBasedPriorRand() {
    final VecBasedMx originalMx = new VecBasedMx(10, new ArrayVec(100));
    for (int i = 0; i < originalMx.rows() - 1; i++) {
      for (int j = 0; j < originalMx.columns(); j++)
        originalMx.set(i, j, rng.nextDouble() < 0.5 && j > 0 ? 1 : 0);
      VecTools.normalizeL1(originalMx.row(i));
    }
    VecTools.fill(originalMx.row(originalMx.rows() - 1), 0);
    final SimplePGM original = new SimplePGM(originalMx);
    checkRestoreFixedTopology(original, PGMEM.POISSON_PRIOR_PATH, 0.5, 100, 0.01);
  }

  private Vec breakV(final Vec next, final double lossProbab) {
    final Vec result = new SparseVec(next.dim());
    final VecIterator it = next.nonZeroes();
    int resIndex = 0;
    while (it.advance()) {
      if (rng.nextDouble() > lossProbab)
        result.set(resIndex++, it.value());
    }
    return result;
  }

  private void checkRestoreFixedTopology(final SimplePGM original, final Computable<ProbabilisticGraphicalModel, PGMEM.Policy> policy, final double lossProbab, final int iterations, final double accuracy) {
    final Vec[] ds = new Vec[100000];
    for (int i = 0; i < ds.length; i++) {
      //TODO: @solar, please, fix it
      final Vec vec = null;
//      do {
//
//      vec = breakV(original.next(rng), lossProbab);
//      }
//      while (VecTools.norm(vec) < MathTools.EPSILON);
      ds[i] = vec;
    }
    final VecDataSet dataSet = new VecDataSetImpl(new RowsVecArrayMx(ds), null);
    final VecBasedMx topology = new VecBasedMx(original.topology.columns(), VecTools.fill(new ArrayVec(original.topology.dim()), 1.));
    VecTools.scale(topology.row(topology.rows() - 1), 0.);
    final PGMEM pgmem = new PGMEM(topology, 0.5, iterations, rng, policy);

    final Action<SimplePGM> listener = new Action<SimplePGM>() {
      int iteration = 0;

      @Override
      public void invoke(final SimplePGM pgm) {
        Interval.stopAndPrint("Iteration " + ++iteration);
        System.out.println();
        System.out.print(VecTools.distance(pgm.topology, original.topology));
        for (int i = 0; i < pgm.topology.columns(); i++) {
          System.out.print(" " + VecTools.distance(pgm.topology.row(i), original.topology.row(i)));
        }
        System.out.println();
        Interval.start();
      }
    };
    pgmem.addListener(listener);

    Interval.start();
    final SimplePGM fit = pgmem.fit(dataSet, new LLLogit(VecTools.fill(new ArrayVec(dataSet.length()), 1.), dataSet));
    VecTools.fill(fit.topology.row(fit.topology.rows() - 1), 0);
    System.out.println(MxTools.prettyPrint(fit.topology));
    System.out.println();
    System.out.println(MxTools.prettyPrint(original.topology));

    assertTrue(VecTools.distance(fit.topology, original.topology) < accuracy * fit.topology.dim());
  }



  public void testElasticNet() {
    {
      final ElasticNetMethod net = new ElasticNetMethod(1e-7f, 0.5, 0);
      final int N = 100;
      final int p = 100;
      Vec beta = new ArrayVec(p);
      for (int i = 0; i < p; ++i) {
        beta.set(i, rng.nextDouble());
      }
      Mx learn = new VecBasedMx(N, p);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < p; ++j)
          learn.set(i, j, rng.nextDouble());
      }
      Vec target = MxTools.multiply(learn, beta);
      Pool pool = new FeaturesTxtPool("fake", new ArraySeq<>(new QURLItem[target.length()]), learn, target);
      final L2 loss = (L2) pool.target(L2.class);
      Linear result = (Linear) net.fit(pool.vecData(), loss);
      assertTrue(VecTools.distance(MxTools.multiply(learn, result.weights), target) < 1e-5f);
      assertTrue(VecTools.distance(beta, result.weights) < 1e-2f);
    }

    //
    {
      final ElasticNetMethod net = new ElasticNetMethod(1e-5f, 0.0, 0.00007);
      final int N = 140;
      final int TestN = 2000;
      final int p = 150;
      Vec beta = new ArrayVec(p);
      for (int i = 0; i < p; ++i) {
        beta.set(i, rng.nextGaussian());
      }
      Mx learn = new VecBasedMx(N, p);
      Mx test = new VecBasedMx(TestN, p);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < p; ++j) {
          learn.set(i, j, rng.nextDouble());
        }
      }
      for (int i = 0; i < TestN; ++i) {
        for (int j = 0; j < p; ++j) {
          test.set(i, j, rng.nextDouble());
        }
      }
      Vec realTarget = MxTools.multiply(learn, beta);
      Vec testTarget = MxTools.multiply(test, beta);
      Vec target = copy(realTarget);
      for (int i=0; i < target.dim();++i) {
        target.adjust(i, rng.nextGaussian()*0.001);
      }
      Pool pool = new FeaturesTxtPool("fake", new ArraySeq<>(new QURLItem[target.length()]), learn, target);
      final L2 loss = (L2) pool.target(L2.class);
      Linear result = (Linear) net.fit(pool.vecData(), loss);
      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
    System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
    System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
    System.out.println("Fit weights " + result.weights);
    System.out.println("Real weights " + beta);
    }


    //check shrinkage
    {
      final ElasticNetMethod net = new ElasticNetMethod(1e-7f, 0.9, 0.005);
      final int N = 100;
      final int NTest = 10000;
      final int p = 500;
      Vec beta = new ArrayVec(p);
      for (int i = 0; i < p; ++i) {
        if (i % 17 == 0) {
          beta.set(i, rng.nextGaussian());
        } else {
          beta.set(i,0);
        }
      }
      Mx learn = new VecBasedMx(N, p);
      Mx test = new VecBasedMx(NTest, p);
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < p; ++j) {
          learn.set(i, j, rng.nextDouble());

        }
      }
      for (int i = 0; i < NTest; ++i) {
        for (int j = 0; j < p; ++j) {
          test.set(i, j, rng.nextDouble());
        }
      }
      Vec realTarget = MxTools.multiply(learn, beta);
      Vec testTarget = MxTools.multiply(test, beta);
      Vec target = copy(realTarget);
      for (int i=0; i < target.dim();++i) {
        target.adjust(i, rng.nextGaussian() * 0.001);
      }
      Pool pool = new FeaturesTxtPool("fake", new ArraySeq<>(new QURLItem[target.length()]), learn, target);
      final L2 loss = (L2) pool.target(L2.class);
      Linear result = (Linear) net.fit(pool.vecData(), loss);
//      for (int i=0; i < beta.dim();++i) {
//        if (beta.get(i) == 0)
//          assertTrue(Math. abs(result.weights.get(i)-0.0) < 1e-9);
//      }
      System.out.println("Learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), realTarget)) / target.dim());
      System.out.println("Noise learn error: " + sqr(VecTools.distance(MxTools.multiply(learn, result.weights), target)) / target.dim());
      System.out.println("Test error: " + sqr(VecTools.distance(MxTools.multiply(test, result.weights), testTarget)) / testTarget.dim());
//      System.out.println("Fit weights " + result.weights);
//      System.out.println("Real weights " + beta);

    }
  }

  public void testLARS() {
    final LARSMethod lars = new LARSMethod();
//    lars.addListener(modelPrinter);
    final Class<L2> targetClass = L2.class;
    final L2 target = learn.target(targetClass);
    final NormalizedLinear model = lars.fit(learn.vecData(), target);
    System.out.println(validate.target(L2.class).value(model.transAll(((VecDataSet) validate).data())));
  }

  public void testGRBoost() {
    final GradientBoosting<L2> boosting = new GradientBoosting<L2>(
            new BootstrapOptimization<L2>(
                    new GreedyRegion(new FastRandom(), GridTools.medianGrid(learn.vecData(), 32)), rng), L2.class, 10000, 0.02);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final L2 target = learn.target(L2.class);
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    final Action modelPrinter = new ModelPrinter();
    final Action qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  public void testGTDRForestBoost() {
    final GradientBoosting<L2> boosting = new GradientBoosting
            (new RegionForest<>(GridTools.medianGrid(learn.vecData(), 32), rng, 5), L2GreedyTDRegion.class, 12000, 0.004);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), learn.target(L2.class));
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(new Action<Trans>() {
      @Override
      public void invoke(final Trans trans) {
        System.out.println();
      }
    });
//    boosting.addListener(modelPrinter);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  public void testGTDRBoost() {
//    final GradientBoosting<L2> boosting = new GradientBoosting
//            (new BootstrapOptimization<>(
//                    new GreedyTDRegion<WeightedLoss<? extends L2>>(GridTools.medianGrid(learn.vecData(), 32)), rng), L2GreedyTDRegion.class, 12000, 0.004);
//    final GradientBoosting<L2> boosting = new GradientBoosting
//            (new RandomForest<>(
//                    new GreedyMergedRegion<WeightedLoss<? extends StatBasedLoss>>(GridTools.medianGrid(learn.vecData(), 32)), rng, 5), L2GreedyTDRegion.class, 12000, 0.004);
    final GradientBoosting<L2> boosting = new GradientBoosting
            (new BootstrapOptimization<>(
                    new GreedyMergedRegion(GridTools.medianGrid(learn.vecData(), 32)), rng), L2GreedyTDRegion.class, 12000, 0.004);
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), learn.target(L2.class));
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    final Action<Trans> counter = new Action<Trans>() {
      int index = 0;

      @Override
      public void invoke(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final Action<Trans> nl = new Action<Trans>() {
      @Override
      public void invoke(final Trans trans) {
        System.out.println();
      }
    };

    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(nl);

    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  public class addBoostingListeners<GlobalLoss extends TargetFunc> {
    addBoostingListeners(final GradientBoosting<GlobalLoss> boosting, final GlobalLoss loss, final Pool<?> _learn, final Pool<?> _validate) {
      final Action counter = new ProgressHandler() {
        int index = 0;

        @Override
        public void invoke(final Trans partial) {
          System.out.print("\n" + index++);
        }
      };
      final ScoreCalcer learnListener = new ScoreCalcer(/*"\tlearn:\t"*/"\t", _learn.vecData(), _learn.target(L2.class));
      final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", _validate.vecData(), _validate.target(L2.class));
      final Action modelPrinter = new ModelPrinter();
      final Action qualityCalcer = new QualityCalcer();
      boosting.addListener(counter);
      boosting.addListener(learnListener);
      boosting.addListener(validateListener);
      boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
      final Ensemble ans = boosting.fit(_learn.vecData(), loss);
      final Vec current = new ArrayVec(_validate.size());
      for (int i = 0; i < _validate.size(); i++) {
        double f = 0;
        for (int j = 0; j < ans.models.length; j++)
          f += ans.weights.get(j) * ((Func) ans.models[j]).value(((VecDataSet) _validate).data().row(i));
        current.set(i, f);
      }
      System.out.println("\n + Final loss = " + VecTools.distance(current, _validate.target(L2.class).target) / Math.sqrt(_validate.size()));

    }
  }

  public void testOTBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn.vecData(), 32), 6), rng), 2000, 0.02);
    new addBoostingListeners<SatL2>(boosting, learn.target(SatL2.class), learn, validate);
  }


  protected static class ScoreCalcer implements ProgressHandler {
    final String message;
    final Vec current;
    private final VecDataSet ds;
    private final L2 target;

    public ScoreCalcer(final String message, final VecDataSet ds, final L2 target) {
      this.message = message;
      this.ds = ds;
      this.target = target;
      current = new ArrayVec(ds.length());
    }

    double min = 1e10;

    @Override
    public void invoke(final Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        for (int i = 0; i < ds.length(); i++) {
          if (increment instanceof Ensemble) {
            current.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
          } else {
            current.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
          }
        }
      } else {
        for (int i = 0; i < ds.length(); i++) {
          current.set(i, ((Func) partial).value(ds.data().row(i)));
        }
      }
      final double curLoss = VecTools.distance(current, target.target) / Math.sqrt(ds.length());
      System.out.print(message + curLoss);
      min = Math.min(curLoss, min);
      System.out.print(" minimum = " + min);
    }
  }

  private static class ModelPrinter implements ProgressHandler {
    @Override
    public void invoke(final Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble model = (Ensemble) partial;
        final Trans increment = model.last();
        System.out.print("\t" + increment);
      }
    }
  }

  private class QualityCalcer implements ProgressHandler {
    Vec residues = VecTools.copy(learn.<L2>target(L2.class).target);
    double total = 0;
    int index = 0;

    @Override
    public void invoke(final Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble model = (Ensemble) partial;
        final Trans increment = model.last();

        final TDoubleIntHashMap values = new TDoubleIntHashMap();
        final TDoubleDoubleHashMap dispersionDiff = new TDoubleDoubleHashMap();
        int index = 0;
        final VecDataSet ds = learn.vecData();
        for (int i = 0; i < ds.data().rows(); i++) {
          final double value;
          if (increment instanceof Ensemble) {
            value = increment.trans(ds.data().row(i)).get(0);
          } else {
            value = ((Func) increment).value(ds.data().row(i));
          }
          values.adjustOrPutValue(value, 1, 1);
          final double ddiff = sqr(residues.get(index)) - sqr(residues.get(index) - value);
          residues.adjust(index, -model.wlast() * value);
          dispersionDiff.adjustOrPutValue(value, ddiff, ddiff);
          index++;
        }
//          double totalDispersion = VecTools.multiply(residues, residues);
        double score = 0;
        for (final double key : values.keys()) {
          final double regularizer = 1 - 2 * Math.log(2) / Math.log(values.get(key) + 1);
          score += dispersionDiff.get(key) * regularizer;
        }
//          score /= totalDispersion;
        total += score;
        this.index++;
        System.out.print("\tscore:\t" + score + "\tmean:\t" + (total / this.index));
      }
    }
  }

  public void testDGraph() {
    final Random rng = new FastRandom();
    for (int n = 1; n < 100; n++) {
      System.out.print("" + n);
      double d = 0;
      for (int t = 0; t < 100000; t++) {
        double sum = 0;
        double sum2 = 0;
        for (int i = 0; i < n; i++) {
          final double v = learn.<L2>target(L2.class).target.get(rng.nextInt(learn.size()));
          sum += v;
          sum2 += v * v;
        }
        d += (sum2 - sum * sum / n) / n;
      }
      System.out.println("\t" + d / 100000);
    }
  }

  public void testFMRun() {
    final FMTrainingWorkaround fm = new FMTrainingWorkaround("r", "1,1,8", "10");
    fm.fit(learn.vecData(), learn.<L2>target(L2.class));
  }
}


