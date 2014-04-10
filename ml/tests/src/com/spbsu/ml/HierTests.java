package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.HierTools;
import com.spbsu.ml.data.impl.Hierarchy;import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.multiclass.MCMacroF1Score;
import com.spbsu.ml.loss.multiclass.MCMacroPrecision;
import com.spbsu.ml.loss.multiclass.MCMacroRecall;
import com.spbsu.ml.loss.multiclass.MCMicroPrecision;
import com.spbsu.ml.loss.multiclass.hier.*;
import com.spbsu.ml.loss.multiclass.hier.impl.HMCMacroF1Score;
import com.spbsu.ml.loss.multiclass.hier.impl.HMCMacroPrecision;
import com.spbsu.ml.loss.multiclass.hier.impl.HMCMacroRecall;
import com.spbsu.ml.loss.multiclass.hier.impl.HMCMicroPrecision;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.HierarchicalClassification;
import com.spbsu.ml.methods.MultiClass;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.array.TDoubleArrayList;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.io.IOException;

/**
 * User: qdeee
 * Date: 07.04.14
 */
public class HierTests extends TestSuite {
  public static Test suite() {
    final TestSuite s = new TestSuite();
    s.addTestSuite(Base.class);
    s.addTestSuite(Regression.class);
    return s;
  }

  public static class Base extends TestCase {
    protected String HIER_XML = "./ml/tests/data/hier/test.xml";
    protected String FEATURES = "./ml/tests/data/hier/test.tsv";

    protected Hierarchy hier;
    protected DataSet learn;
    protected DataSet test;

    protected int minEntries = 3;
    protected int iters = 10;
    protected double weakStep = 0.04;

    protected void defaultLoad(String hierXml, String featuresTxt) throws IOException {
      hier = HierTools.loadHierarchicalStructure(hierXml);
      learn = DataTools.loadFromFeaturesTxt(featuresTxt);
    }

    @Override
    protected void setUp() throws Exception {
      defaultLoad(HIER_XML, FEATURES);
    }

    public void testLoading() {
      hier.fill(learn);
      Hierarchy.traversePrint(hier.getRoot());
    }

    public void testPruning()  {
      hier.fill(learn);
      Hierarchy prunedTree = hier.getPrunedCopy(minEntries);
      Hierarchy.traversePrint(prunedTree.getRoot());
    }

    public void testFitting() {
      HierarchicalClassification classification = new HierarchicalClassification(iters, weakStep);
      HierLoss mainLoss = new HMCMacroPrecision(hier, learn, minEntries);
      HierLoss[] learnLosses = new HierLoss[] {
          new HMCMicroPrecision(mainLoss, learn.target()),
          mainLoss,
          new HMCMacroRecall(mainLoss, learn.target()),
          new HMCMacroF1Score(mainLoss, learn.target())
      };
      HierarchicalModel model = (HierarchicalModel) classification.fit(learn, mainLoss);

      Vec learnPredicted = model.bestClassAll(learn.data());
      for (int i = 0; i < learnLosses.length; i++) {
        double val = learnLosses[i].value(learnPredicted);
        System.out.println("[LEARN] metric: " + learnLosses[i].getClass().getSimpleName() + ", value = " + val);
      }
      if (test != null) {
        HierLoss[] testLosses = new HierLoss[] {
            new HMCMicroPrecision(mainLoss, test.target()),
            new HMCMacroPrecision(mainLoss, test.target()),
            new HMCMacroRecall(mainLoss, test.target()),
            new HMCMacroF1Score(mainLoss, test.target())
        };
        Vec testPredicted = model.bestClassAll(test.data());
        for (int i = 0; i < testLosses.length; i++) {
          double val = testLosses[i].value(testPredicted);
          System.out.println("[TEST] metric: " + testLosses[i].getClass().getSimpleName() + ", value = " + val);
        }
      }
    }

  }
  public static class Regression extends Base {
    private static final String FEATURES_TEST = "./ml/tests/data/featuresTest.txt.gz";

    private int depth = 4;

    @Override
    protected void setUp() throws Exception {
      FEATURES = "./ml/tests/data/features.txt.gz";

      depth = 4;
      minEntries = 450;
      weakStep = 1.5;
      iters = 1000;

      TDoubleArrayList borders = new TDoubleArrayList();
      learn = HierTools.loadRegressionAsMC(FEATURES, 1 << depth, borders);
      test = HierTools.loadRegressionAsMC(FEATURES_TEST, 1 << depth, borders);
      hier = HierTools.prepareHierStructForRegressionMedian(learn.target());
      //    hier = prepareHierStructForRegressionUniform(depth);
      //    VecTools.append(learn.target(), VecTools.fill(new ArrayVec(learn.power()), (1 << depth) - 1));
    }
  }

  public static class Something extends TestCase {
    public void testBaseline() throws IOException {
      final double weakStep = 0.5;
      final int iters = 1;
      final TDoubleArrayList borders = new TDoubleArrayList();
      final int classCount = 16;
      final DataSet learn = HierTools.loadRegressionAsMC("./ml/tests/data/features.txt.gz", classCount, borders);
      final DataSet test = HierTools.loadRegressionAsMC("./ml/tests/data/featuresTest.txt.gz", classCount, borders);
      final BFGrid grid = GridTools.medianGrid(learn, 32);
      final MLLLogit learnLoss = new MLLLogit(learn.target());
      final MLLLogit testLoss = new MLLLogit(test.target());

      final GradientBoosting<MLLLogit> boosting = new GradientBoosting<MLLLogit>(new MultiClass(new GreedyObliviousTree(grid, 5),
          new Computable<Vec, L2>() {
            @Override
            public L2 compute(Vec argument) {
              return new SatL2(argument);
            }
          }
      ), iters, weakStep);
      final ProgressHandler calcer = new ProgressHandler() {
        int index = 0;

        @Override
        public void invoke(Trans partial) {
          if ((index + 1) % 20 == 0) {
            double lvalue = learnLoss.value(partial.transAll(learn.data()));
            double tvalue = testLoss.value(partial.transAll(test.data()));
            System.out.println("iter=" + index + ", [learn]MLLLogitValue=" + lvalue + ", [test]MLLLogitValue=" + tvalue);
          }
          index++;
        }
      };
      boosting.addListener(calcer);
      Ensemble ensemble = boosting.fit(learn, learnLoss);
      MultiClassModel model = HierarchicalClassification.joinBoostingResults(ensemble);
      final Func.Stub learnP = new MCMicroPrecision(learn.target());
      final Func.Stub testP = new MCMicroPrecision(test.target());
      double learnPVal = learnP.value(model.bestClassAll(learn.data()));
      double testPVal = testP.value(model.bestClassAll(test.data()));
      System.out.println("learnPVal = " + learnPVal);
      System.out.println("testPVal = " + testPVal);
    }
  }
}