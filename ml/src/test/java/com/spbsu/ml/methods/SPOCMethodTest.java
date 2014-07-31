package com.spbsu.ml.methods;

import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.*;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multiclass.MCMacroF1Score;
import com.spbsu.ml.loss.multiclass.MCMacroPrecision;
import com.spbsu.ml.loss.multiclass.MCMacroRecall;
import com.spbsu.ml.loss.multiclass.MCMicroPrecision;
import com.spbsu.ml.meta.impl.FakeTargetMeta;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.methods.spoc.AbstractCodingMatrixLearning;
import com.spbsu.ml.methods.spoc.CMLMetricOptimization;
import com.spbsu.ml.methods.spoc.SPOCMethodClassic;
import com.spbsu.ml.methods.spoc.impl.CodingMatrixLearning;
import com.spbsu.ml.methods.spoc.impl.CodingMatrixLearningGreedy;
import com.spbsu.ml.methods.spoc.impl.CodingMatrixLearningGreedyParallels;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.test_utils.TestResourceLoader;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
* User: qdeee
* Date: 07.05.14
*/
public abstract class SPOCMethodTest extends TestSuite {

  private abstract static class Base extends TestCase {
    protected Pool<?> learn;
    protected Pool<?> test;
    protected Mx S;

    protected int k;
    protected int l;

    protected double lambdaC;
    protected double lambdaR;
    protected double lambda1;
    protected double mxStep;

    protected double mcStep;
    protected int mcIters;

    protected double metricC;
    protected int metricIters;
    protected double metricStep;

    protected static void evalModel(final MultiClassModel model, final Pool ds, final String prefixComment) {
      final Vec predict = model.bestClassAll(ds.vecData().data());
      final Func[] metrics = new Func[] {
          ds.target(MCMicroPrecision.class),
          ds.target(MCMacroPrecision.class),
          ds.target(MCMacroRecall.class),
          ds.target(MCMacroF1Score.class),
      };
      for (Func metric : metrics) {
        double val = metric.value(predict);
        System.out.println(prefixComment + metric.getClass().getSimpleName() + ", value = " + val);
      }
    }

    public void testBaseline() {
      final BlockwiseMLLLogit learnLoss = (BlockwiseMLLLogit) learn.target(BlockwiseMLLLogit.class);
      final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
      final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), SatL2.class), mcIters, mcStep);
      final ProgressHandler listener = new ProgressHandler() {
        int iter = 0;
        @Override
        public void invoke(final Trans partial) {
          if ((iter+1) % 20 == 0) {
            final MultiClassModel model = MultiClassModel.joinBoostingResults((Ensemble) partial);
            evalModel(model, learn, iter + "[LEARN] ");
            evalModel(model, test, iter + "[TEST] ");
            System.out.println();
          }
          iter++;
        }
      };
      boosting.addListener(listener);
      final Ensemble ensemble = boosting.fit(learn.vecData(), learnLoss);
      final MultiClassModel model = MultiClassModel.joinBoostingResults(ensemble);
      evalModel(model, learn, "[LEARN] ");
      evalModel(model, test, "[TEST] ");
      System.out.println();
    }

    public void _testSimilarityMatrix() {
      final BlockwiseMLLLogit target = (BlockwiseMLLLogit) learn.target(BlockwiseMLLLogit.class);
      final Mx similarityMatrix = MCTools.createSimilarityMatrix(learn.vecData(), target.labels());
      System.out.println(similarityMatrix.toString());
    }

    public void testFindMx() {
      final AbstractCodingMatrixLearning codingMatrixLearning = getCodingMatrixLearning();
      final Mx codingMatrix = codingMatrixLearning.trainCodingMatrix(S);
      System.out.println(codingMatrix.toString());
      if (!CodingMatrixLearning.checkConstraints(codingMatrix)) {
        throw new IllegalStateException("Result is out of constraints");
      }
    }

    public void testFit() {
      final AbstractCodingMatrixLearning cml = getCodingMatrixLearning();
      final Mx codingMatrix = cml.trainCodingMatrix(S);
//      if (!CodingMatrixLearning.checkConstraints(codingMatrix)) {
//        throw new IllegalStateException("Result matrix is out of constraints");
//      }

      final VecOptimization method = new SPOCMethodClassic(codingMatrix, mcStep, mcIters);
      final MultiClassModel model = (MultiClassModel) method.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
      evalModel(model, learn, "[LEARN] ");
      evalModel(model, test, "[TEST] ");
      System.out.println();
    }

    protected AbstractCodingMatrixLearning getCodingMatrixLearning() {
      return new CodingMatrixLearning(k, l, mxStep, lambdaC, lambdaR, lambda1);
    }

    public void _testFitWithProbs() {
      final AbstractCodingMatrixLearning cml = getCodingMatrixLearning();
      final Mx codingMatrix = cml.trainCodingMatrix(S);
      if (!CodingMatrixLearning.checkConstraints(codingMatrix)) {
        throw new IllegalStateException("Result matrix is out of constraints");
      }

      final VecOptimization<BlockwiseMLLLogit> method = new SPOCMethodClassic(codingMatrix, mcStep, mcIters);
      final MultiClassModel model = (MultiClassModel) method.fit(learn.vecData(), (BlockwiseMLLLogit) learn.target(BlockwiseMLLLogit.class));
      evalModel(model, learn, "[LEARN] ");
      evalModel(model, test, "[TEST] ");

      final CMLMetricOptimization metricOptimization = new CMLMetricOptimization(learn.vecData(), (BlockwiseMLLLogit) learn.target(BlockwiseMLLLogit.class),S, metricC, metricStep);
      final Mx mu = metricOptimization.trainProbs(codingMatrix, model.dirs());
      System.out.println(mu.toString());
    }
  }



  public abstract static class DefaultDataTests extends Base {
    private final static double[] hierBorders = new double[] {0.038125, 0.07625, 0.114375, 0.1525, 0.61};
    public static final double[] classicBorders = new double[]{0.06999, 0.13999, 0.40999, 0.60999, 0.61};

    public void setUp() throws Exception {
      super.setUp();
      final TDoubleList borders = new TDoubleArrayList();
//      borders.addAll(classicBorders);
      borders.addAll(hierBorders);
      learn = TestResourceLoader.loadPool("features.txt.gz");
      test = TestResourceLoader.loadPool("featuresTest.txt.gz");
      final IntSeq learnTarget = MCTools.transformRegressionToMC(learn.target(L2.class).target, borders.size(), borders);
      final IntSeq testTarget = MCTools.transformRegressionToMC(test.target(L2.class).target, borders.size(), borders);
      learn.addTarget(new FakeTargetMeta(learn.vecData(), FeatureMeta.ValueType.INTS), learnTarget);
      test.addTarget(new FakeTargetMeta(test.vecData(), FeatureMeta.ValueType.INTS), testTarget);

//      final CharSequence mxStr = StreamTools.readStream(TestResourceLoader.loadResourceAsStream("multiclass/regression_based/features.txt.similarityMx"));
      final CharSequence mxStr = StreamTools.readStream(TestResourceLoader.loadResourceAsStream("multiclass/regression_based/features-simmatrix-classic.txt"));
      S = MathTools.CONVERSION.convert(mxStr, Mx.class);

      k = borders.size();
      l = 5;

      lambdaC = 3.0;
      lambdaR = 2.5;
      lambda1 = 7.0;
      mxStep = 1.8;

      mcIters = 100;
      mcStep = 0.3;

      metricC = 0.5;
      metricIters = 100;
      metricStep = 0.3;
    }

    @Override
    public void testBaseline() {
      mcIters = 100;
      mcStep = 0.5;
      super.testBaseline();
    }
  }

  public abstract static class GreedyDefaultDS extends DefaultDataTests {
    @Override
    public void setUp() throws Exception {
      super.setUp();
      mcIters = 200;
    }

    @Override
    protected AbstractCodingMatrixLearning getCodingMatrixLearning() {
      return new CodingMatrixLearningGreedy(k, l, lambdaC, lambdaR, lambda1);
    }
  }

  public abstract static class ParallelGreedyDefaultDS extends DefaultDataTests {
    @Override
    protected AbstractCodingMatrixLearning getCodingMatrixLearning() {
      return new CodingMatrixLearningGreedyParallels(k, l, lambdaC, lambdaR, lambda1);
    }
  }

//  public abstract static class ExternDataTests extends Base {
//    public void setUp() throws Exception {
//      super.setUp();
//      final VecDataSet fullds = DataTools.loadFakePool("./ml/tests/data/multiclass/ds_letter/letter.libfm");
//      final Pair<DataSet, DataSet> pairDS = MCTools.splitCvMulticlass(fullds, 0.8, new FastRandom(100501));
//      learn = pairDS.first;
//      test = pairDS.second;
//      S = loadMxFromFile("./ml/tests/data/multiclass/ds_letter/letter.similarityMx");
//
//      k = S.rows();
//      l = 10;
//
//      lambdaC = 10.0;
//      lambdaR = 2.5;
//      lambda1 = 5.0;
//      mxStep = 1.8;
//
//      mcIters = 300;
//      mcStep = 0.7;
//
//      metricC = 1.0;
//      metricStep = 0.3;
//      metricIters = 100;
//    }
//
//  }


  public void testCreateConstraintsMatrix() throws Exception {
    final Mx B = new VecBasedMx(
        2,
        new ArrayVec(
            1, -1,
            -1,  1,
            0,  1)
    );
    final Mx constraintsMatrix = CodingMatrixLearning.createConstraintsMatrix(B);
    System.out.println("constraints matrix:");
    System.out.println(constraintsMatrix.toString());
  }

}
