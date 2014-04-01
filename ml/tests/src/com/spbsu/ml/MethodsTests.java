package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.SparseVec;
import com.spbsu.commons.math.vectors.impl.VecArrayMx;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.NormalizedLinear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.trees.GreedyContinuesObliviousSoftBondariesRegressionTree;
import com.spbsu.ml.methods.trees.GreedyExponentialObliviousTree;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.ContinousObliviousTree;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.models.PolynomialExponentRegion;
import com.spbsu.ml.models.ProbabilisticGraphicalModel;
import gnu.trove.map.hash.TDoubleDoubleHashMap;
import gnu.trove.map.hash.TDoubleIntHashMap;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Random;
import java.util.Scanner;

import static com.spbsu.commons.math.MathTools.sqr;

/**
 * User: solar
 * Date: 26.11.12
 * Time: 15:50
 */
public class MethodsTests extends GridTest {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public void testPGMFit3x3() {
    ProbabilisticGraphicalModel original = new ProbabilisticGraphicalModel(new VecBasedMx(3, new ArrayVec(new double[]{
            0, 0.2, 0.8,
            0, 0, 1.,
            0, 0, 0
    })));
    checkRestoreFixedTopology(original, PGMEM.MOST_PROBABLE_PATH, 0.0, 10, 0.01);
  }

  public void testPGMFit5x5() {
    ProbabilisticGraphicalModel original = new ProbabilisticGraphicalModel(new VecBasedMx(5, new ArrayVec(new double[]{
            0, 0.2, 0.3,  0.1,  0.4,
            0, 0,   0.25, 0.25, 0.5,
            0, 0,   0,    0.1,  0.9,
            0, 0,   0.5,  0,    0.5,
            0, 0,   0,    0,    0
    })));


    checkRestoreFixedTopology(original, PGMEM.MOST_PROBABLE_PATH, 0., 10, 0.01);
  }

  public void testPGMFit5x5RandSkip() {
    final ProbabilisticGraphicalModel original = new ProbabilisticGraphicalModel(new VecBasedMx(5, new ArrayVec(new double[]{
            0, 0.2, 0.3,  0.1,  0.4,
            0, 0,   0.25, 0.25, 0.5,
            0, 0,   0,    0.1,  0.9,
            0, 0,   0.5,  0,    0.5,
            0, 0,   0,    0,    0
    })));

    checkRestoreFixedTopology(original, PGMEM.LAPLACE_PRIOR_PATH, 0.8, 100, 0.05);
  }
  public void testPGMFit10x10Rand() {
    final VecBasedMx originalMx = new VecBasedMx(10, new ArrayVec(100));
    for (int i = 0; i < originalMx.rows() - 1; i++) {
      for (int j = 0; j < originalMx.columns(); j++)
        originalMx.set(i, j, rng.nextDouble() < 0.5 && j > 0 ? 1 : 0);
      VecTools.normalizeL1(originalMx.row(i));
    }
    VecTools.fill(originalMx.row(originalMx.rows() - 1), 0);
    ProbabilisticGraphicalModel original = new ProbabilisticGraphicalModel(originalMx);
    checkRestoreFixedTopology(original, PGMEM.LAPLACE_PRIOR_PATH, 0.5, 100, 0.01);
  }

  private Vec breakV(Vec next, double lossProbab) {
    Vec result = new SparseVec<IntBasis>(new IntBasis(next.dim()));
    final VecIterator it = next.nonZeroes();
    int resIndex = 0;
    while (it.advance()) {
      if (rng.nextDouble() > lossProbab)
        result.set(resIndex++, it.value());
    }
    return result;
  }

  private void checkRestoreFixedTopology(final ProbabilisticGraphicalModel original, Factory<PGMEM.Policy> policy, double lossProbab, int iterations, double accuracy) {
    Vec[] ds = new Vec[10000];
    for (int i = 0; i < ds.length; i++) {
      Vec vec;
//      do {
      vec = breakV(original.next(rng), lossProbab);
//      }
//      while (VecTools.norm(vec) < MathTools.EPSILON);
      ds[i] = vec;
    }
    final DataSetImpl dataSet = new DataSetImpl(new VecArrayMx(ds), new ArrayVec(ds.length));
    final PGMEM pgmem = new PGMEM(new VecBasedMx(original.topology.columns(), VecTools.fill(new ArrayVec(original.topology.dim()), 1.)), 0.2, iterations, rng, policy);
    final Action<ProbabilisticGraphicalModel> listener = new Action<ProbabilisticGraphicalModel>() {
      @Override
      public void invoke(ProbabilisticGraphicalModel pgm) {
        System.out.print(VecTools.distance(pgm.topology, original.topology));
        for (int i = 0; i < pgm.topology.columns(); i++) {
          System.out.print(" " + VecTools.distance(pgm.topology.row(i), original.topology.row(i)));
        }
        System.out.println();
      }
    };
    pgmem.addListener(listener);

    final ProbabilisticGraphicalModel fit = pgmem.fit(dataSet, new LLLogit(VecTools.fill(new ArrayVec(dataSet.power()), 1.)));
    VecTools.fill(fit.topology.row(fit.topology.rows() - 1), 0);
    System.out.println(VecTools.prettyPrint(fit.topology));
    System.out.println();
    System.out.println(VecTools.prettyPrint(original.topology));

    assertTrue(VecTools.distance(fit.topology, original.topology) < accuracy * fit.topology.dim());
  }

  public void testLARS() {
    final LARSMethod lars = new LARSMethod();
//    lars.addListener(modelPrinter);
    final NormalizedLinear model = lars.fit(learn, new L2(learn.target()));
    System.out.println(new L2(validate.target()).value(model.transAll(validate.data())));
  }

  public void testGRBoost() {
    final GradientBoosting<L2> boosting = new GradientBoosting<L2>(new BootstrapOptimization<L2>(new GreedyRegion(new FastRandom(), GridTools.medianGrid(learn, 32)), rng),
      new Computable<Vec, L2>() {
        @Override
        public L2 compute(Vec argument) {
          return new L2(argument);
        }
      }, 10000, 0.02);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate);
    final Action modelPrinter = new ModelPrinter();
    final Action qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn, new L2(learn.target()));
  }

  public void testGTDRBoost() {
    final GradientBoosting<L2> boosting = new GradientBoosting<L2>(new BootstrapOptimization<L2>(new GreedyTDRegion<WeightedLoss<L2>>(GridTools.medianGrid(learn, 32)), rng), 10000, 0.02);
    final Action counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate);
    final Action modelPrinter = new ModelPrinter();
    final Action qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn, new L2(learn.target()));
  }

  public class addBoostingListeners<GlobalLoss extends Func> {
    addBoostingListeners(GradientBoosting<GlobalLoss> boosting, GlobalLoss loss, DataSet _learn, DataSet _validate) {
      final Action counter = new ProgressHandler() {
        int index = 0;

        @Override
        public void invoke(Trans partial) {
          System.out.print("\n" + index++);
        }
      };
      final ScoreCalcer learnListener = new ScoreCalcer(/*"\tlearn:\t"*/"\t", _learn);
      final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", _validate);
      final Action modelPrinter = new ModelPrinter();
      final Action qualityCalcer = new QualityCalcer();
      boosting.addListener(counter);
      boosting.addListener(learnListener);
      boosting.addListener(validateListener);
      boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
      final Ensemble ans = boosting.fit(_learn, loss);
      Vec current = new ArrayVec(_validate.power());
      for (int i = 0; i < _validate.power(); i++) {
        double f = 0;
        for (int j = 0; j < ans.models.length; j++)
          f += ans.weights.get(j) * ((Func)ans.models[j]).value(_validate.data().row(i));
        current.set(i, f);
      }
      System.out.println("\n + Final loss = " + VecTools.distance(current, _validate.target()) / Math.sqrt(_validate.power()));

    }
  }

  public void testOTBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn, 32), 6), rng), 2000, 0.02);
    new addBoostingListeners<SatL2>(boosting, new SatL2(learn.target()), learn, validate);
  }


  protected static class ScoreCalcer implements ProgressHandler {
    final String message;
    final Vec current;
    private final DataSet ds;

    public ScoreCalcer(String message, DataSet ds) {
      this.message = message;
      this.ds = ds;
      current = new ArrayVec(ds.power());
    }

    double min = 1e10;

    @Override
    public void invoke(Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        final DSIterator iter = ds.iterator();
        int index = 0;
        while (iter.advance()) {
          current.adjust(index++, linear.wlast() * ((Func) increment).value(iter.x()));
        }
      } else {
        final DSIterator iter = ds.iterator();
        int index = 0;
        while (iter.advance()) {
          current.set(index++, ((Func) partial).value(iter.x()));
        }
      }
      double curLoss = VecTools.distance(current, ds.target()) / Math.sqrt(ds.power());
      System.out.print(message + curLoss);
      min = Math.min(curLoss, min);
      System.out.print(" minimum = " + min);
    }
  }

  private static class ModelPrinter implements ProgressHandler {
    @Override
    public void invoke(Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble model = (Ensemble) partial;
        final Trans increment = model.last();
        System.out.print("\t" + increment);
      }
    }
  }

  private class QualityCalcer implements ProgressHandler {
    Vec residues = VecTools.copy(learn.target());
    double total = 0;
    int index = 0;

    @Override
    public void invoke(Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble model = (Ensemble) partial;
        final Trans increment = model.last();

        final DSIterator iterator = learn.iterator();
        final TDoubleIntHashMap values = new TDoubleIntHashMap();
        final TDoubleDoubleHashMap dispersionDiff = new TDoubleDoubleHashMap();
        int index = 0;
        while (iterator.advance()) {
          final double value = ((Func) increment).value(iterator.x());
          values.adjustOrPutValue(value, 1, 1);
          final double ddiff = sqr(residues.get(index)) - sqr(residues.get(index) - value);
          residues.adjust(index, -model.wlast() * value);
          dispersionDiff.adjustOrPutValue(value, ddiff, ddiff);
          index++;
        }
//          double totalDispersion = VecTools.multiply(residues, residues);
        double score = 0;
        for (double key : values.keys()) {
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
    Random rng = new FastRandom();
    for (int n = 1; n < 100; n++) {
      System.out.print("" + n);
      double d = 0;
      for (int t = 0; t < 100000; t++) {
        double sum = 0;
        double sum2 = 0;
        for (int i = 0; i < n; i++) {
          double v = learn.target().get(rng.nextInt(learn.power()));
          sum += v;
          sum2 += v * v;
        }
        d += (sum2 - sum * sum / n) / n;
      }
      System.out.println("\t" + d / 100000);
    }
  }
  
  public void testFMRun() {
    FMTrainingWorkaround fm = new FMTrainingWorkaround("r", "1,1,8", "10");
    fm.fit(learn, null);
  }
}


