package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.HierTools;
import com.spbsu.ml.data.impl.Hierarchy;
import com.spbsu.ml.loss.hier.*;
import com.spbsu.ml.methods.HierarchicalClassification;
import com.spbsu.ml.models.HierarchicalModel;
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
      HierLoss mainLoss = new MCMacroPrecision(hier, learn, minEntries);
      HierLoss[] learnLosses = new HierLoss[] {
          new MCMicroPrecision(mainLoss, learn.target()),
          new MCMicroRecall(mainLoss, learn.target()),
          new MCMicroF1Score(mainLoss, learn.target()),
          mainLoss,
          new MCMacroRecall(mainLoss, learn.target()),
          new MCMacroF1Score(mainLoss, learn.target())
      };
      HierarchicalModel model = (HierarchicalModel) classification.fit(learn, mainLoss);

      Vec learnPredicted = model.bestClassAll(learn.data());
      for (int i = 0; i < learnLosses.length; i++) {
        double val = learnLosses[i].value(learnPredicted);
        System.out.println("[LEARN] metric: " + learnLosses[i].getClass().getSimpleName() + ", value = " + val);
      }
      if (test != null) {
        HierLoss[] testLosses = new HierLoss[] {
            new MCMicroPrecision(mainLoss, test.target()),
            new MCMicroRecall(mainLoss, test.target()),
            new MCMicroF1Score(mainLoss, test.target()),
            new MCMacroPrecision(mainLoss, test.target()),
            new MCMacroRecall(mainLoss, test.target()),
            new MCMacroF1Score(mainLoss, test.target())
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
      minEntries = 3;
      weakStep = 1.5;
      iters = 10;

      TDoubleArrayList borders = new TDoubleArrayList();
      learn = HierTools.loadRegressionAsMC(FEATURES, 1 << depth, borders);
      test = HierTools.loadRegressionAsMC(FEATURES_TEST, 1 << depth, borders);
      hier = HierTools.prepareHierStructForRegressionMedian(learn.target());
      //    hier = prepareHierStructForRegressionUniform(depth);
      //    VecTools.append(learn.target(), VecTools.fill(new ArrayVec(learn.power()), (1 << depth) - 1));
    }
  }
}