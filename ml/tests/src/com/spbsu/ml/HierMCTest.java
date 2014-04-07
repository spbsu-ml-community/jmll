package com.spbsu.ml;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.data.impl.Hierarchy;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.*;
import com.spbsu.ml.loss.hier.*;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.HierarchicalClassification;
import com.spbsu.ml.methods.MultiClass;
import com.spbsu.ml.methods.Optimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.array.TDoubleArrayList;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;

import java.io.IOException;

/**
 * User: qdeee
 * Date: 03.03.14
 */

@RunWith(Enclosed.class)
public class HierMCTest{

  @Ignore
  public static class Base {
    protected static Hierarchy hier;
    protected static DataSet learn;
    protected static DataSet test;
    protected static int minEntries = 3;
    protected static int iters = 10;
    protected static double weakStep = 0.04;

    protected static void defaultLoad(String hierXml, String featuresTxt) throws IOException {
      hier = DataTools.loadHierarchicalStructure(hierXml);
      learn = DataTools.loadFromFeaturesTxt(featuresTxt);
    }

    @Test
    public void loading() {
      hier.fill(learn);
      Hierarchy.traversePrint(hier.getRoot());
    }

    @Test
    public void pruning()  {
      hier.fill(learn);
      Hierarchy prunedTree = hier.getPrunedCopy(minEntries);
      Hierarchy.traversePrint(prunedTree.getRoot());
    }

    @Test
    public void fitting() {
      HierarchicalClassification classification = new HierarchicalClassification(iters, weakStep, minEntries);
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

  public static class SimpleMC extends Base {
    private static final String HIER_XML = "./ml/tests/data/hier/test.xml";
    private static final String FEATURES = "./ml/tests/data/hier/test.tsv";

    @BeforeClass
    public static void onStart() throws IOException {
      defaultLoad(HIER_XML, FEATURES);
    }

    @Test
    public void fitting() {
      super.fitting();
    }
  }

  public static class MediumMC extends Base {
    private static final String HIER_XML = "./ml/tests/data/hier/hier.xml";
    private static final String FEATURES = "./ml/tests/data/hier/hmc3.txt.gz";

    @BeforeClass
    public static void onStart() throws IOException {
      defaultLoad(HIER_XML, FEATURES);
      minEntries = 100;
    }

    @Ignore
    @Test
    public void outPrunedXml() throws IOException {
      Hierarchy prunedCopy = hier.getPrunedCopy(minEntries);
      DataTools.saveHierarchicalStructure(prunedCopy, HIER_XML + ".pruned");
    }
  }

  public static class Regression extends Base {
    private static final String FEATURES = "./ml/tests/data/features.txt.gz";
    private static final String FEATURES_TEST = "./ml/tests/data/featuresTest.txt.gz";

    @BeforeClass
    public static void onStart() throws IOException {
      int depth = 4;
      minEntries = 450;
      weakStep = 1;
      iters = 100;

      TDoubleArrayList borders = new TDoubleArrayList();
      learn = loadRegressionAsMC(FEATURES, 1 << depth, borders);
      test = loadRegressionAsMC(FEATURES_TEST, 1 << depth, borders);
      hier = prepareHierStructForRegressionMedian(learn.target());
//      hier = prepareHierStructForRegressionUniform(depth);
//      VecTools.append(learn.target(), VecTools.fill(new ArrayVec(learn.power()), (1 << depth) - 1));
    }

    @Test
    public void fitting() {
      super.fitting();
    }

    @Test
    public void baselineMC() throws IOException {
      DataSet ds = DataTools.normalizeClasses(learn);
      BFGrid grid = GridTools.medianGrid(ds, 32);
      Optimization<MLLLogit> optimization = new GradientBoosting<MLLLogit>(new MultiClass(new GreedyObliviousTree(grid, 5), new Computable<Vec, L2>() {
        @Override
        public L2 compute(Vec argument) {
          return new SatL2(argument);
        }
      }), 100, 0.04);
      MLLLogit loss = new MLLLogit(ds.target());
      Ensemble<MultiClassModel> model = (Ensemble<MultiClassModel>) optimization.fit(ds, loss);
      double value = loss.value(model.transAll(ds.data()));
      System.out.println("value = " + value);
    }

    private static Hierarchy prepareHierStructForRegressionUniform(int depth)  {
      Hierarchy.CategoryNode root = new Hierarchy.CategoryNode(0, null);

      Hierarchy.CategoryNode[] nodes = new Hierarchy.CategoryNode[(1 << (depth + 1)) - 1];
      nodes[0] = root;
      for (int i = 1; i < nodes.length; i++) {
        Hierarchy.CategoryNode parent = nodes[i % 2 == 0 ? (i - 2) / 2 : (i - 1) / 2];
        nodes[i] = new Hierarchy.CategoryNode(i, parent);
        parent.addChild(nodes[i]);
      }
      return new Hierarchy(root);
    }

    private static class Counter {
      int number = 0;

      public Counter(int init) {this.number = init;}
      public int getNext()     {return number++;}
    }

    private static Hierarchy prepareHierStructForRegressionMedian(Vec targetMC) {
      double[] target = targetMC.toArray();
      int clsCount = DataTools.countClasses(targetMC);
      int[] freq = new int[clsCount];
      for (int i = 0; i < target.length; i++) {
        freq[(int)target[i]]++;
      }
      Hierarchy.CategoryNode root = splitAndAddChildren(freq, 0, freq.length, new Counter(freq.length));
      return new Hierarchy(root);

    }

    private static Hierarchy.CategoryNode splitAndAddChildren(int[] arr, int start, int end, Counter innerNodeIdx) {
      int sum = 0;
      for (int i = start; i < end; i++) {
        sum += arr[i];
      }

      int bestSplit = -1;
      int minSubtract = Integer.MAX_VALUE;
      int curSum = 0;
      for (int split = start; split < end - 1; split++) {
        curSum += arr[split];
        int subtract = Math.abs((sum - curSum) - curSum);
        if (subtract < minSubtract) {
          minSubtract = subtract;
          bestSplit = split;
        }
      }

      Hierarchy.CategoryNode node = new Hierarchy.CategoryNode(innerNodeIdx.getNext(), null);
      if (bestSplit == start) {
        node.addChild(new Hierarchy.CategoryNode(start, node));
      }
      else {
        node.addChild(splitAndAddChildren(arr, start, bestSplit + 1, innerNodeIdx));
      }
      if (bestSplit == end - 2) {
        node.addChild(new Hierarchy.CategoryNode(end - 1, node));
      }
      else {
        node.addChild(splitAndAddChildren(arr, bestSplit + 1, end, innerNodeIdx));
      }
      return node;
    }

    private static DataSet loadRegressionAsMC(String file, int classCount, TDoubleArrayList borders)  throws IOException{
      DataSet ds = DataTools.loadFromFeaturesTxt(file);

      double[] target = ds.target().toArray();
      int[] idxs = ArrayTools.sequence(0, target.length);
      ArrayTools.parallelSort(target, idxs);

      if (borders.size() == 0) {
        double min = target[0];
        double max = target[target.length - 1];
        double delta = (max - min) / classCount;
        for (int i = 0; i < classCount; i++) {
          borders.add(delta * (i + 1));
        }
      }

      Vec resultTarget = new ArrayVec(ds.power());
      int targetCursor = 0;
      for (int borderCursor = 0; borderCursor < borders.size(); borderCursor++){
        while (targetCursor < target.length && target[targetCursor] <= borders.get(borderCursor)) {
          resultTarget.set(idxs[targetCursor], borderCursor);
          targetCursor++;
        }
      }
      return new ChangedTarget((DataSetImpl)ds, resultTarget);
    }
  }

}