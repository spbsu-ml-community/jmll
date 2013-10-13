package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.trees.GreedyContinuesObliviousSoftBondariesRegressionTree;
import com.spbsu.ml.methods.trees.GreedyObliviousRegressionTree;
import com.spbsu.ml.models.*;
import gnu.trove.TDoubleDoubleHashMap;
import gnu.trove.TDoubleIntHashMap;

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
    rng = new FastRandom();
  }

  public void testLARS() {
    final LARSMethod boosting = new LARSMethod();
//    boosting.addProgressHandler(modelPrinter);
    final NormalizedLinearModel model = boosting.fit(learn, new L2Loss(learn.target()));
    System.out.println(new L2Loss(validate.target()).value(model.value(validate)));
  }

  public void testGRBoost() {
    final GradientBoosting boosting = new GradientBoosting(new GreedyRegion(new FastRandom(), learn, GridTools.medianGrid(learn, 32)), 10000, 0.02, rng);
    final ProgressHandler counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void progress(Model partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate);
    final ProgressHandler modelPrinter = new ModelPrinter();
    final ProgressHandler qualityCalcer = new QualityCalcer();
    boosting.addProgressHandler(counter);
    boosting.addProgressHandler(learnListener);
    boosting.addProgressHandler(validateListener);
    boosting.addProgressHandler(qualityCalcer);
//    boosting.addProgressHandler(modelPrinter);
    boosting.fit(learn, new L2Loss(learn.target()));
  }

  public void testGRSBoost() {
    final GradientBoosting boosting = new GradientBoosting(new GreedyL1SphereRegion(new FastRandom(), learn, GridTools.medianGrid(learn, 32)), 10000, 0.02, rng);
    final ProgressHandler counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void progress(Model partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate);
    final ProgressHandler modelPrinter = new ModelPrinter();
    final ProgressHandler qualityCalcer = new QualityCalcer();
    boosting.addProgressHandler(counter);
    boosting.addProgressHandler(learnListener);
    boosting.addProgressHandler(validateListener);
    boosting.addProgressHandler(qualityCalcer);
//    boosting.addProgressHandler(modelPrinter);
    boosting.fit(learn, new L2Loss(learn.target()));
  }

  public void testGTDRBoost() {
    final GradientBoosting boosting = new GradientBoosting(new GreedyTDRegion(new FastRandom(), learn, GridTools.medianGrid(learn, 32)), 10000, 0.02, rng);
    final ProgressHandler counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void progress(Model partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate);
    final ProgressHandler modelPrinter = new ModelPrinter();
    final ProgressHandler qualityCalcer = new QualityCalcer();
    boosting.addProgressHandler(counter);
    boosting.addProgressHandler(learnListener);
    boosting.addProgressHandler(validateListener);
    boosting.addProgressHandler(qualityCalcer);
//    boosting.addProgressHandler(modelPrinter);
    boosting.fit(learn, new L2Loss(learn.target()));
  }

  public void testOTBoost() {
    final GradientBoosting boosting = new GradientBoosting(new GreedyObliviousRegressionTree(new FastRandom(), learn, GridTools.medianGrid(learn, 32), 6), 2000, 0.005, rng);
    final ProgressHandler counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void progress(Model partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate);
    final ProgressHandler modelPrinter = new ModelPrinter();
    final ProgressHandler qualityCalcer = new QualityCalcer();
    boosting.addProgressHandler(counter);
    boosting.addProgressHandler(learnListener);
    boosting.addProgressHandler(validateListener);
    boosting.addProgressHandler(qualityCalcer);
//    boosting.addProgressHandler(modelPrinter);
    boosting.fit(learn, new L2Loss(learn.target()));
  }

  public void testCOTBoost() {
    final GradientBoosting boosting = new GradientBoosting(new GreedyContinuesObliviousSoftBondariesRegressionTree(new FastRandom(), learn, GridTools.medianGrid(learn, 32), 6), 2000, 0.003, rng);
    final ProgressHandler counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void progress(Model partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate);
    final ProgressHandler modelPrinter = new ModelPrinter();
    final ProgressHandler qualityCalcer = new QualityCalcer();
    boosting.addProgressHandler(counter);
    boosting.addProgressHandler(learnListener);
    boosting.addProgressHandler(validateListener);
    boosting.addProgressHandler(qualityCalcer);
//    boosting.addProgressHandler(modelPrinter);
    boosting.fit(learn, new L2Loss(learn.target()));
  }

  private double sqr(double x) {
    return x * x;
  }

  public void testTreeBoost() {
    final GradientBoosting boosting = new GradientBoosting(new BestAtomicSplitMethod(), 1000, 0.01, rng);
    final ScoreCalcer learnListener = new ScoreCalcer("learn : ", learn);
    final ScoreCalcer validateListener = new ScoreCalcer(" test : ", validate);
    final ProgressHandler modelPrinter = new ProgressHandler() {
      @Override
      public void progress(Model partial) {
        if (partial instanceof AdditiveModel) {
          final AdditiveModel model = (AdditiveModel) partial;
          System.out.println("\n" + model.models.get(model.models.size() - 1));
        }
      }
    };
    boosting.addProgressHandler(modelPrinter);
    boosting.addProgressHandler(learnListener);
    boosting.addProgressHandler(validateListener);
    boosting.fit(learn, new L2Loss(learn.target()));
  }

  public void testObliviousTree() {
    ScoreCalcer scoreCalcerValidate = new ScoreCalcer(" On validate data Set loss = ", validate);
    ScoreCalcer scoreCalcerLearn = new ScoreCalcer(" On learn data Set loss = ", learn);
    for (int depth = 1; depth <= 6; depth++) {
      ObliviousTree tree = (ObliviousTree) new GreedyObliviousRegressionTree(new FastRandom(), learn, GridTools.medianGrid(learn, 32), depth).fit(learn, new L2Loss(learn.target()));
      System.out.print("Oblivious Tree depth = " + depth);
      scoreCalcerLearn.progress(tree);
      scoreCalcerValidate.progress(tree);
      System.out.println();
    }
  }

  public void testContinousObliviousTree() {
    ScoreCalcer scoreCalcerValidate = new ScoreCalcer(" On validate data Set loss = ", validate);
    ScoreCalcer scoreCalcerLearn = new ScoreCalcer(" On learn data Set loss = ", learn);
    for (int depth = 1; depth <= 6; depth++) {
      ContinousObliviousTree tree = new GreedyContinuesObliviousSoftBondariesRegressionTree(new FastRandom(), learn, GridTools.medianGrid(learn, 32), depth).fit(learn, new L2Loss(learn.target()));
      //for(int i = 0; i < 10/*learn.target().dim()*/;i++)
      // System.out.println(learn.target().get(i) + "= " + tree.value(learn.data().row(i)));
      System.out.print("Oblivious Tree deapth = " + depth);
      scoreCalcerLearn.progress(tree);
      scoreCalcerValidate.progress(tree);

      System.out.println();
    }
  }

  public void testDebugContinousObliviousTree() {
    //ScoreCalcer scoreCalcerValidate = new ScoreCalcer(" On validate data Set loss = ", validate);
    double[] data = {0, 1, 2};
    double[] target = {0, 1, 2};

    DataSet debug = new DataSetImpl(data, target);
    ScoreCalcer scoreCalcerLearn = new ScoreCalcer(" On learn data Set loss = ", debug);
    for (int depth = 1; depth <= 1; depth++) {
      ContinousObliviousTree tree = new GreedyContinuesObliviousSoftBondariesRegressionTree(new FastRandom(), debug, GridTools.medianGrid(debug, 32), depth).fit(debug, new L2Loss(debug.target()));
      System.out.print("Oblivious Tree deapth = " + depth);
      scoreCalcerLearn.progress(tree);
      System.out.println();
    }
  }

  private static class ScoreCalcer implements ProgressHandler {
    final String message;
    final Vec current;
    private final DataSet ds;

    public ScoreCalcer(String message, DataSet ds) {
      this.message = message;
      this.ds = ds;
      current = new ArrayVec(ds.power());
    }

    @Override
    public void progress(Model partial) {
      if (partial instanceof AdditiveModel) {
        final AdditiveModel additiveModel = (AdditiveModel) partial;
        final Model increment = (Model) additiveModel.models.get(additiveModel.models.size() - 1);
        final DSIterator iter = ds.iterator();
        int index = 0;
        while (iter.advance()) {
          current.adjust(index++, additiveModel.step * increment.value(iter.x()));
        }
      } else {
        final DSIterator iter = ds.iterator();
        int index = 0;
        while (iter.advance()) {
          current.set(index++, partial.value(iter.x()));
        }
      }
      System.out.print(message + VecTools.distance(current, ds.target()) / Math.sqrt(ds.power()));
    }
  }

  private static class ModelPrinter implements ProgressHandler {
    @Override
    public void progress(Model partial) {
      if (partial instanceof AdditiveModel) {
        final AdditiveModel model = (AdditiveModel) partial;
        final Model increment = (Model) model.models.get(model.models.size() - 1);
        System.out.print("\t" + increment);
      }
    }
  }

  private class QualityCalcer implements ProgressHandler {
    Vec residues = VecTools.copy(learn.target());
    double total = 0;
    int index = 0;

    @Override
    public void progress(Model partial) {
      if (partial instanceof AdditiveModel) {
        final AdditiveModel model = (AdditiveModel) partial;
        final Model increment = (Model) model.models.get(model.models.size() - 1);

        final DSIterator iterator = learn.iterator();
        final TDoubleIntHashMap values = new TDoubleIntHashMap();
        final TDoubleDoubleHashMap dispersionDiff = new TDoubleDoubleHashMap();
        int index = 0;
        while (iterator.advance()) {
          final double value = increment.value(iterator.x());
          values.adjustOrPutValue(value, 1, 1);
          final double ddiff = sqr(residues.get(index)) - sqr(residues.get(index) - value);
          residues.adjust(index, -model.step * value);
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

//    private int getIndexI(int N, int n) {
//        int tale = n*(n+1)/2 - N - 1;
//        return n - (int)( (Math.sqrt(8*tale + 1) - 1) / 2) - 1;
//    }
//
//    private int getIndexJ(int N, int n) {
//        return N-getN(getIndexI(N,n), 0, n);
//    }

    private int getN(int i, int j, int n) {
        return n*(n+1)/2 - (n-i+1)*(n-i)/2 + j-i;
//        return i*(2*n - i - 1)/2 + j;
    }


    public void testQuadraticRegression() {
        double r = 3;
        int N = 8;
        int n = 2;

        Model expectedModel = new QuadraticModel(new VecBasedMx(n, new ArrayVec(4, 0,
                0, 3)),
                new ArrayVec(1, 2), 0.0);

        Mx data = new VecBasedMx(N, n);
        for (int i = 0; i < N; i++) {
            data.set(i, 0, r * Math.cos(2 * Math.PI * i / N));
            data.set(i, 1, r * Math.sin(2 * Math.PI * i / N));
        }
        Vec target = new ArrayVec(N);
        for (int i = 0; i < N; i++) {
            target.set(i, expectedModel.value(data.row(i)));
        }
        DataSet ds = new DataSetImpl(data, target);

        System.out.println("expected model: \n" + expectedModel.toString());
        System.out.println("expected model values: " + expectedModel.value(ds).toString());

        MLMethodOrder1 quadrRegr = new LeastSquaresQuadratic();
        Model model = quadrRegr.fit(ds, new L2Loss(target));
        System.out.println("actual model: \n" + model.toString());
        System.out.println();
        System.out.println("actual model values: " + model.value(ds).toString());
    }
  }
}


