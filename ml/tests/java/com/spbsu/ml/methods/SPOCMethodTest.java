package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.types.TypeConverter;
import com.spbsu.commons.math.io.Mx2CharSequenceConversionPack;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.*;
import com.spbsu.ml.clustering.ClusterizationAlgorithm;
import com.spbsu.ml.clustering.impl.KMeansAlgorithm;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.multiclass.MCMacroF1Score;
import com.spbsu.ml.loss.multiclass.MCMacroPrecision;
import com.spbsu.ml.loss.multiclass.MCMacroRecall;
import com.spbsu.ml.loss.multiclass.MCMicroPrecision;
import com.spbsu.ml.methods.spoc.*;
import com.spbsu.ml.methods.spoc.impl.CodingMatrixLearning;
import com.spbsu.ml.methods.spoc.impl.CodingMatrixLearningGreedy;
import com.spbsu.ml.methods.spoc.impl.CodingMatrixLearningGreedyParallels;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.map.TIntObjectMap;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.io.*;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class SPOCMethodTest extends TestSuite {

  private static class Base extends TestCase {
    protected DataSet learn;
    protected DataSet test;
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

    protected static void evalModel(final MultiClassModel model, final DataSet ds, final String prefixComment) {
      final Vec predict = model.bestClassAll(ds.data());
      final Func[] metrics = new Func[] {
          new MCMicroPrecision(ds.target()),
          new MCMacroPrecision(ds.target()),
          new MCMacroRecall(ds.target()),
          new MCMacroF1Score(ds.target())
      };
      for (Func metric : metrics) {
        double val = metric.value(predict);
        System.out.println(prefixComment + metric.getClass().getSimpleName() + ", value = " + val);
      }
    }

    public void testBaseline() {
      final MLLLogit learnLoss = new MLLLogit(learn.target());
      final BFGrid grid = GridTools.medianGrid(learn, 32);
      final GradientBoosting<MLLLogit> boosting = new GradientBoosting<MLLLogit>(new MultiClass(new GreedyObliviousTree(grid, 5),
          new Computable<Vec, L2>() {
            @Override
            public L2 compute(Vec argument) {
              return new SatL2(argument);
            }
          }
      ), mcIters, mcStep);
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
      final Ensemble ensemble = boosting.fit(learn, learnLoss);
      final MultiClassModel model = MultiClassModel.joinBoostingResults(ensemble);
      evalModel(model, learn, "[LEARN] ");
      evalModel(model, test, "[TEST] ");
      System.out.println();
    }

    public void _testSimilarityMatrix() {
      final Mx similarityMatrix = MCTools.createSimilarityMatrix(learn);
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

      final Optimization<LLLogit> method = new SPOCMethodClassic(codingMatrix, mcStep, mcIters);
      final MultiClassModel model = (MultiClassModel) method.fit(learn, null);
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

      final Optimization<LLLogit> method = new SPOCMethodClassic(codingMatrix, mcStep, mcIters);
      final MultiClassModel model = (MultiClassModel) method.fit(learn, null);
      evalModel(model, learn, "[LEARN] ");
      evalModel(model, test, "[TEST] ");

      final CMLMetricOptimization metricOptimization = new CMLMetricOptimization(learn, S, metricC, metricIters, metricStep);
      final Mx mu = metricOptimization.trainProbs(codingMatrix, model.dirs());
      System.out.println(mu.toString());
    }
  }

  public static class DefaultDataTests extends Base {
    private final static double[] hierBorders = new double[] {0.038125, 0.07625, 0.114375, 0.1525, 0.61};
    public static final double[] classicBorders = new double[]{0.06999, 0.13999, 0.40999, 0.60999, 0.61};

    public void setUp() throws Exception {
      super.setUp();
      final TDoubleList borders = new TDoubleArrayList();
//      borders.addAll(new double[] {0.038125, 0.07625, 0.114375, 0.1525, 0.61});
      borders.addAll(classicBorders);
//      borders.addAll(hierBorders);
      learn = MCTools.loadRegressionAsMC("./ml/tests/data/features.txt.gz", borders.size(), borders);
      test = MCTools.loadRegressionAsMC("./ml/tests/data/featuresTest.txt.gz", borders.size(), borders);
//      S = loadMxFromFile("./ml/tests/data/multiclass/regression_based/features.txt.similarityMx");
      S = loadMxFromFile("/Users/qdeee/Datasets/features-simmatrix-classic.txt");

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

  public static class GreedyDefaultDS extends DefaultDataTests {

    //      lambdaC = 3;
    //      lambdaR = 2.5;
    //      lambda1 = 2;
    @Override
    protected AbstractCodingMatrixLearning getCodingMatrixLearning() {
      return new CodingMatrixLearningGreedy(k, l, lambdaC, lambdaR, lambda1);
    }
  }

  public static class ParallelGreedyDefaultDS extends DefaultDataTests {
    @Override
    protected AbstractCodingMatrixLearning getCodingMatrixLearning() {
      return new CodingMatrixLearningGreedyParallels(k, l, lambdaC, lambdaR, lambda1);
    }
  }

  public static class ExternDataTests extends Base {
    public void setUp() throws Exception {
      super.setUp();
      final DataSet fullds = DataTools.loadFromSparseFeaturesTxt("./ml/tests/data/multiclass/ds_letter/letter.libfm");
      final Pair<DataSet, DataSet> pairDS = MCTools.splitCvMulticlass(fullds, 0.8, new FastRandom(100501));
      learn = pairDS.first;
      test = pairDS.second;
      S = loadMxFromFile("./ml/tests/data/multiclass/ds_letter/letter.similarityMx");

      k = S.rows();
      l = 10;

      lambdaC = 10.0;
      lambdaR = 2.5;
      lambda1 = 5.0;
      mxStep = 1.8;

      mcIters = 300;
      mcStep = 0.7;

      metricC = 1.0;
      metricStep = 0.3;
      metricIters = 100;
    }

  }

  public static class StuffTests extends TestCase{

    private DataSet learn;

    @Override
    protected void setUp() throws Exception {
      learn = DataTools.loadFromFeaturesTxt("./ml/tests/data/features.txt.gz");
    }

    public void testRelevanceSplit() throws IOException {
      final DataSet dataSet = MCTools.normalizeClasses(learn);
      final TIntObjectMap<TIntList> indexes = MCTools.splitClassesIdxs(dataSet);
      for (TIntObjectIterator<TIntList> iter = indexes.iterator(); iter.hasNext(); ) {
        iter.advance();
        System.out.println(iter.key() + " : " + iter.value().size());
      }
    }

    public void testHistSplit() throws Exception{

    }
  }


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

  private static void writeMxToFile(final Mx mx, final String filename) throws FileNotFoundException {
    final PrintWriter writer = new PrintWriter(new BufferedOutputStream(new FileOutputStream(filename)));
    final TypeConverter<Mx, CharSequence> converter = new Mx2CharSequenceConversionPack.Mx2CharSequenceConverter();
    final CharSequence mxStr = converter.convert(mx);
    writer.write(mxStr.toString());
    writer.flush();
  }

  private static Mx loadMxFromFile(final String filename) throws IOException {
    final TypeConverter<CharSequence, Mx> converter = new Mx2CharSequenceConversionPack.CharSequence2MxConverter();
    final BufferedReader reader = new BufferedReader(new FileReader(new File(filename)));
    final StringBuilder builder = new StringBuilder();
    String s;
    while ((s = reader.readLine()) != null) {
      builder.append(s);
      builder.append("\n");
    }
    return converter.convert(builder.toString());
  }
}
