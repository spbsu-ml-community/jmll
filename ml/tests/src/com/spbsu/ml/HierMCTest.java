package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.data.impl.HierDS;
import com.spbsu.ml.methods.HierarchicalClassification;
import com.spbsu.ml.models.HierarchicalModel;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;

import java.io.IOException;
import java.util.Random;

/**
 * User: qdeee
 * Date: 03.03.14
 */

@RunWith(Enclosed.class)
public class HierMCTest{

  @Ignore
  public static class Base {
    protected static HierDS hier;
    protected static int minEntries = 10;
    protected static int iters = 1;

    protected static void defaultLoad(String hierXml, String featuresTxt) throws IOException {
      hier = DataTools.loadHierarchicalStructure(hierXml);
      DataSet features = DataTools.loadFromFeaturesTxt(featuresTxt);
      hier.fill(features);
    }

    @Test
    public void loading() {
      HierDS.traversePrint(hier.getRoot());
    }

    @Test
    public void pruning()  {
      HierDS prunedTree = hier.getPrunedCopy(minEntries);
      HierDS.traversePrint(prunedTree.getRoot());
    }

    @Test
    public void fitting() {
      HierDS prunedTree = hier.getPrunedCopy(minEntries);
      HierarchicalClassification classification = new HierarchicalClassification(iters, 0.04);
      HierarchicalModel fit = (HierarchicalModel) classification.fit(prunedTree, null);
    }
  }

  public static class SimpleMC extends Base {
    private static final String HIER_XML = "./ml/tests/data/hier/test.xml";
    private static final String FEATURES = "./ml/tests/data/hier/test.tsv";

    @BeforeClass
    public static void onStart() throws IOException {
      defaultLoad(HIER_XML, FEATURES);
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

    @Test
    public void outPrunedXml() throws IOException {
      HierDS prunedCopy = hier.getPrunedCopy(minEntries);
      DataTools.saveHierarchicalStructure(prunedCopy, HIER_XML + ".pruned");
    }
  }

  public static class Regression extends Base {
    private static final String FEATURES = "./ml/tests/data/features.txt.gz";

    @BeforeClass
    public static void onStart() throws IOException {
      int depth = 4;
      hier = prepareHierStructForRegression(depth);
      DataSet features = loadRegressionAsMC(FEATURES, depth, new TDoubleArrayList());
      VecTools.append(features.target(), VecTools.fill(new ArrayVec(features.power()), (1 << depth) - 1));
      hier.fill(features);
    }

    private static HierDS prepareHierStructForRegression(int depth) throws IOException {
      HierDS.CategoryNode root = new HierDS.CategoryNode(0, null);

      HierDS.CategoryNode[] nodes = new HierDS.CategoryNode[(1 << (depth + 1)) - 1];
      nodes[0] = root;
      for (int i = 1; i < nodes.length; i++) {
        HierDS.CategoryNode parent = nodes[i % 2 == 0 ? (i - 2) / 2 : (i - 1) / 2];
        nodes[i] = new HierDS.CategoryNode(i, parent);
        parent.addChild(nodes[i]);
      }
      return new HierDS(root);
    }

    private static DataSet loadRegressionAsMC(String file, int depth, TDoubleArrayList borders)  throws IOException{
      DataSet ds = DataTools.loadFromFeaturesTxt(file);

      double[] target = ds.target().toArray();
      int[] idxs = ArrayTools.sequence(0, target.length);
      ArrayTools.parallelSort(target, idxs);
      double min = target[0];
      double max = target[target.length - 1];
      double delta = (max - min) / (1 << depth);

      if (borders.size() == 0) {
        int count = 1 << depth;
        for (int i = 0; i < count; i++) {
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

  private static Pair<DataSet, DataSet> splitCV(DataSet learn, double percent, Random rnd) {
    TIntObjectHashMap<TIntList> catId2Idxs = new TIntObjectHashMap<TIntList>();
    TIntArrayList learnIndices = new TIntArrayList();
    TIntArrayList testIndices = new TIntArrayList();

    for (DSIterator iter = learn.iterator(); iter.advance(); ) {
      int catId = (int) iter.y();
      if (catId2Idxs.containsKey(catId)) {
        catId2Idxs.get(catId).add(iter.index());
      }
      else {
        TIntList idxs = new TIntLinkedList();
        idxs.add(iter.index());
        catId2Idxs.put(catId, idxs);
      }
    }

    for (TIntObjectIterator<TIntList> iterator = catId2Idxs.iterator(); iterator.hasNext(); ) {
      iterator.advance();
      TIntList idxs = iterator.value();
      idxs.shuffle(rnd);
      int split = (int) (idxs.size() * percent);
      for (int i = 0; i < split; i++)
        learnIndices.add(idxs.get(i));
      for (int i = split; i < idxs.size(); i++)
        testIndices.add(idxs.get(i));
    }

    final int[] learnIndicesArr = learnIndices.toArray();
    final int[] testIndicesArr = testIndices.toArray();
    return Pair.<DataSet, DataSet>create(
        new DataSetImpl(
            new VecBasedMx(
                learn.xdim(),
                new IndexTransVec(
                    learn.data(),
                    new RowsPermutation(learnIndicesArr, learn.xdim())
                )
            ),
            new IndexTransVec(learn.target(), new ArrayPermutation(learnIndicesArr))
        ),
        new DataSetImpl(
            new VecBasedMx(
                learn.xdim(),
                new IndexTransVec(
                    learn.data(),
                    new RowsPermutation(testIndicesArr, learn.xdim())
                )
            ),
            new IndexTransVec(learn.target(), new ArrayPermutation(testIndicesArr))));
  }
}