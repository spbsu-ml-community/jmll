package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.HierTools;
import com.spbsu.ml.data.impl.HierarchyTree;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.multiclass.MCMicroPrecision;
import com.spbsu.ml.loss.multiclass.hier.*;
import com.spbsu.ml.loss.multiclass.hier.impl.HMCMacroF1Score;
import com.spbsu.ml.loss.multiclass.hier.impl.HMCMacroPrecision;
import com.spbsu.ml.loss.multiclass.hier.impl.HMCMacroRecall;
import com.spbsu.ml.loss.multiclass.hier.impl.HMCMicroPrecision;
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MCModel;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.linked.TIntLinkedList;
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
    s.addTestSuite(BaseRefinement.class);
    return s;
  }

  public static class Base extends TestCase {
    protected String HIER_XML = "./ml/tests/data/hier/test.xml";
    protected String FEATURES = "./ml/tests/data/hier/test1.tsv";

    protected HierarchyTree hier;
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
      HierarchyTree.traversePrint(hier.getRoot());
    }

    public void testPruning()  {
      hier.fill(learn);
      HierarchyTree prunedTree = hier.getPrunedCopy(minEntries);
      HierarchyTree.traversePrint(prunedTree.getRoot());
    }

    protected Trans fit(HierLoss mainLoss) {
      HierarchicalClassification classification = new HierarchicalClassification(iters, weakStep);
      return classification.fit(learn, mainLoss);
    }

    public void testFitting() {
      HierLoss mainLoss = new HMCMacroPrecision(hier, learn, minEntries);

      double time = System.currentTimeMillis();
      MCModel model = (MCModel) fit(mainLoss);
      System.out.println("Learning time: " + ((System.currentTimeMillis() - time) / 1000) + " sec");
      HierLoss[] learnLosses = new HierLoss[] {
          new HMCMicroPrecision(mainLoss, learn.target()),
          mainLoss,
          new HMCMacroRecall(mainLoss, learn.target()),
          new HMCMacroF1Score(mainLoss, learn.target())
      };

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
      iters = 100;

      TDoubleArrayList borders = new TDoubleArrayList();
      learn = HierTools.loadRegressionAsMC(FEATURES, 1 << depth, borders);
      test = HierTools.loadRegressionAsMC(FEATURES_TEST, 1 << depth, borders);
      hier = HierTools.prepareHierStructForRegressionMedian(learn.target());
      //    hier = prepareHierStructForRegressionUniform(depth);
      //    VecTools.append(learn.target(), VecTools.fill(new ArrayVec(learn.power()), (1 << depth) - 1));
    }
  }

  public static class Baseline extends TestCase {
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
      MCModel model = MCModel.joinBoostingResults(ensemble);
      final Func.Stub learnP = new MCMicroPrecision(learn.target());
      final Func.Stub testP = new MCMicroPrecision(test.target());
      double learnPVal = learnP.value(model.bestClassAll(learn.data()));
      double testPVal = testP.value(model.bestClassAll(test.data()));
      System.out.println("learnPVal = " + learnPVal);
      System.out.println("testPVal = " + testPVal);
    }
  }

  public static class BaseRefinement extends Base {
    @Override
    protected Trans fit(final HierLoss mainLoss) {
      HierarchicalRefinedClassification classification = new HierarchicalRefinedClassification(iters, weakStep);
      return classification.fit(learn, mainLoss);
    }
  }

  public static class RegressionRefinement extends Regression {
    @Override
    protected Trans fit(final HierLoss mainLoss) {
      HierarchicalRefinedClassification classification = new HierarchicalRefinedClassification(iters, weakStep);
      return classification.fit(learn, mainLoss);
    }
  }

  public static class Stuff extends TestCase {
    public void testSt() {
      Vec vec = new ArrayVec(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
      Vec indVec = new IndexTransVec(vec,
          new ArrayPermutation(new int[] {8, 1}));
      for (int i = 0; i < indVec.dim(); i++) {
        System.out.println(indVec.get(i));
      }
    }

    public void testSt2() {
      TIntList list = new TIntLinkedList();
      final int i = list.indexOf(5);
      System.out.println(i);
    }

    public void testSt3() {
      System.out.println("lala");
      for (int i = 0; i < 10; i++) {
        System.out.print(i + "\r");
      }
      System.out.println("lala");
    }

    public void testSt4() {
      int i, N=20;
      for (i = 0; i < N; i--) {
        System.out.print("*");
      }
    }

  }

}