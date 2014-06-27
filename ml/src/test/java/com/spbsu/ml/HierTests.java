package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.HierTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.data.impl.HierarchyTree;
import com.spbsu.ml.func.Ensemble;
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
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.hierarchical.HierarchicalClassification;
import com.spbsu.ml.methods.hierarchical.HierarchicalRefinedClassification;
import com.spbsu.ml.methods.hierarchical.HierarchicalRefinedExpertClassification;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierJoinedBinClassAddMetaFeaturesModel;
import com.spbsu.ml.models.HierRefinedExpertModel;
import com.spbsu.ml.models.MCModel;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.array.TDoubleArrayList;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.io.IOException;

/**
 * User: qdeee
 * Date: 07.04.14
 */
public class HierTests extends TestSuite {
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

    public void testCopyStructure() {
      hier.fill(learn);
      final HierarchyTree prunedCopy = hier.getPrunedCopy(minEntries);
      final HierarchyTree structureCopy = prunedCopy.getStructureCopy();
      HierarchyTree.traversePrint(structureCopy.getRoot());
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
      iters = 200;

      TDoubleArrayList borders = new TDoubleArrayList();
      learn = MCTools.loadRegressionAsMC(FEATURES, 1 << depth, borders);
      test = MCTools.loadRegressionAsMC(FEATURES_TEST, 1 << depth, borders);
      System.out.println(borders.toString());
      hier = HierTools.prepareHierStructForRegressionMedian(learn.target());
      //    hier = prepareHierStructForRegressionUniform(depth);
      //    VecTools.append(learn.target(), VecTools.fill(new ArrayVec(learn.power()), (1 << depth) - 1));
    }
  }

  public static class Baseline extends TestCase {
    public void testBaseline() throws IOException {
      final double weakStep = 1.5;
      final int iters = 1000;
      final TDoubleArrayList borders = new TDoubleArrayList(new double[]{0.038125, 0.07625, 0.114375, 0.1525, 0.61});
      final int classCount = 5;
      final DataSet learn = MCTools.loadRegressionAsMC("./ml/tests/data/features.txt.gz", classCount, borders);
      final DataSet test = MCTools.loadRegressionAsMC("./ml/tests/data/featuresTest.txt.gz", classCount, borders);
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
      MCModel model = MultiClassModel.joinBoostingResults(ensemble);

      Func.Stub[] learnLosses = new Func.Stub[] {
          new MCMicroPrecision(learn.target()),
          new MCMacroPrecision(learn.target()),
          new MCMacroRecall(learn.target()),
          new MCMacroF1Score(learn.target())
      };

      Vec learnPredicted = model.bestClassAll(learn.data());
      for (int i = 0; i < learnLosses.length; i++) {
        double val = learnLosses[i].value(learnPredicted);
        System.out.println("[LEARN] metric: " + learnLosses[i].getClass().getSimpleName() + ", value = " + val);
      }
      if (test != null) {
        Func.Stub[] testLosses = new Func.Stub[] {
            new MCMicroPrecision(test.target()),
            new MCMacroPrecision(test.target()),
            new MCMacroRecall(test.target()),
            new MCMacroF1Score(test.target())
        };
        Vec testPredicted = model.bestClassAll(test.data());
        for (int i = 0; i < testLosses.length; i++) {
          double val = testLosses[i].value(testPredicted);
          System.out.println("[TEST] metric: " + testLosses[i].getClass().getSimpleName() + ", value = " + val);
        }
      }
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

  public static class BaseRefinementExperts extends Base {
    @Override
    protected Trans fit(final HierLoss mainLoss) {
      HierarchicalRefinedExpertClassification classification = new HierarchicalRefinedExpertClassification(iters, weakStep);
      return classification.fit(learn, mainLoss);
    }
  }

  public static class RegressionRefinimentExperts extends Regression {
    private void calcMetaFeaturesStats(final HierJoinedBinClassAddMetaFeaturesModel btModel, final HierLoss origLoss) {
      final HierarchyTree learnTree = hier.getStructureCopy();
      final HierarchyTree testTree = hier.getStructureCopy();

      final Vec mappedLearnTarget = new ArrayVec(learn.power());
      for (int i = 0; i < mappedLearnTarget.dim(); i++) {
        double val = learn.target().get(i);
        double mappedVal = origLoss.targetMapping.get((int) val);
        mappedLearnTarget.set(i, mappedVal);
      }
      final Vec mappedTestTarget = new ArrayVec(test.power());
      for (int i = 0; i < mappedTestTarget.dim(); i++) {
        double val = test.target().get(i);
        double mappedVal = origLoss.targetMapping.get((int)val);
        mappedTestTarget.set(i, mappedVal);
      }

      learnTree.fill(new ChangedTarget((DataSetImpl) learn, mappedLearnTarget));
      testTree.fill(new ChangedTarget((DataSetImpl) test, mappedTestTarget));

      traverseCalcMetaFeaturesStats(btModel, learnTree.getRoot(), testTree.getRoot());
    }

    private void traverseCalcMetaFeaturesStats(final HierJoinedBinClassAddMetaFeaturesModel btModel, final HierarchyTree.Node learnNode, final HierarchyTree.Node testNode) {
      for (int i = 0; i < learnNode.getChildren().size(); i++) {
        final HierarchyTree.Node childLearnNode = learnNode.getChildren().get(i);
        final HierarchyTree.Node childTestNode = testNode.getChildren().get(i);
        if (childLearnNode.isLeaf())
          continue;
        final int label = childLearnNode.getCategoryId();
        final HierJoinedBinClassAddMetaFeaturesModel childModel = btModel.getModelByLabel(label);
        if (childModel != null) {
          final DataSet learnNodeDSForChild = learnNode.createDSForChild(label);
          final DataSet testNodeDSForChild = testNode.createDSForChild(label);
          final Mx learnChildProbs = childModel.probsAll(learnNodeDSForChild.data());
          final Mx testChildProbs = childModel.probsAll(testNodeDSForChild.data());
          final Vec[] learnColumns = MxTools.splitMxColumns(learnChildProbs);
          final Vec[] testColumns = MxTools.splitMxColumns(testChildProbs);
          for (int j = 0; j < learnColumns.length; j++) {
            Vec learnColumn = learnColumns[j];
            Vec testColumn = testColumns[j];
            System.out.println();
            HierTools.printMeanAndVarForClassificationOut(learnNodeDSForChild.target(), learnColumn, "[LEARN] Calc stats for " + childModel.classLabels.get(j));
            System.out.println();
            HierTools.printMeanAndVarForClassificationOut(testNodeDSForChild.target(), testColumn, "[TEST] Calc stats for " + childModel.classLabels.get(j));
          }
          traverseCalcMetaFeaturesStats(childModel, childLearnNode, childTestNode);
        }
      }
    }


    @Override
    protected Trans fit(final HierLoss mainLoss) {
      HierarchicalRefinedExpertClassification classification = new HierarchicalRefinedExpertClassification(iters, weakStep);
      final HierRefinedExpertModel model = (HierRefinedExpertModel) classification.fit(learn, mainLoss);
      calcMetaFeaturesStats(model.bottomUpModel, mainLoss);
      return model;
    }
  }
}