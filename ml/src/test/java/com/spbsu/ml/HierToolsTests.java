package com.spbsu.ml;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.tree.FastTree;
import com.spbsu.commons.util.tree.Tree;
import com.spbsu.ml.data.tools.HierTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.FakeTargetMeta;
import com.spbsu.ml.test_utils.TestResourceLoader;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.procedure.TIntIntProcedure;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import junit.framework.TestCase;

/**
 * User: qdeee
 * Date: 25.07.14
 */
public class HierToolsTests extends TestCase {
  protected static Pool<?> pool;
  protected static IntSeq smallTarget;
  protected static IntSeq bigTarget;

  private synchronized void init() throws Exception {
    if (pool == null || smallTarget == null || bigTarget == null) {
      pool = TestResourceLoader.loadPool("features.txt.gz");
      smallTarget = MCTools.transformRegressionToMC(pool.target(L2.class).target, 5, new TDoubleArrayList(new double[]{0.038125, 0.07625, 0.114375, 0.1525, 0.61}));
      bigTarget = MCTools.transformRegressionToMC(pool.target(L2.class).target, 16, new TDoubleArrayList());
    }
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    init();
  }

  public void testHierBuilder() throws Exception {
    final FastTree tree = HierTools.loadOrderedMulticlassAsHierarchicalMedian(smallTarget);
    System.out.println(tree.toString());
  }

  public void testHierBuilder2() throws Exception {
    final FastTree tree = HierTools.loadOrderedMulticlassAsHierarchicalMedian(bigTarget);
    final FastTree goodTree = HierTools._loadOrderedMulticlassAsHierarchicalMedian(bigTarget);

    final String actual = tree.toString();
    final String expected = goodTree.toString();
    System.out.println(actual);
    System.out.println("\n\n");
    System.out.println(expected);

    assertNotSame(expected, actual);
  }

  public void testCounter() throws Exception {
    final FastTree tree = HierTools.loadOrderedMulticlassAsHierarchicalMedian(smallTarget);
    final TIntIntMap id2deepCount = HierTools.itemsDeepCounter(tree, smallTarget);
    System.out.println(id2deepCount.toString());

  }

  public void testPruning() throws Exception {
    final IntSeq intTarget = MCTools.transformRegressionToMC(pool.target(L2.class).target, 16, new TDoubleArrayList());
    final FastTree tree = HierTools.loadOrderedMulticlassAsHierarchicalMedian(intTarget);
    final Tree pruneTree = HierTools.pruneTree(tree, intTarget, 450);
    System.out.println(pruneTree.toString());

  }

  public void testMapping() throws Exception {
    final IntSeq intTarget = MCTools.transformRegressionToMC(pool.target(L2.class).target, 16, new TDoubleArrayList());
    final FastTree tree = HierTools._loadOrderedMulticlassAsHierarchicalMedian(intTarget);
    final Tree pruneTree = HierTools.pruneTree(tree, intTarget, 450);
    final TIntIntMap map = new TIntIntHashMap();
    HierTools.createTreesMapping(tree.getRoot(), pruneTree.getRoot(), map);
    final TIntSet unprunedNodes = new TIntHashSet(new int[]{0, 1, 2, 3, 16, 17, 18, 19, 20});
    map.forEachEntry(new TIntIntProcedure() {
      @Override
      public boolean execute(final int k, final int v) {
        if (unprunedNodes.contains(k)) {
          assertEquals(k, v);
        } else {
          assertEquals(20, v);
        }
        return true;
      }
    });
  }
}
