package com.spbsu.ml;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.tree.IntArrayTree;
import com.spbsu.commons.util.tree.IntTree;
import com.spbsu.ml.data.tools.HierTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.L2;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntIntMap;

/**
 * User: qdeee
 * Date: 25.07.14
 */
public class HierToolsTests extends GridTest {
  public void testHierBuilder3() throws Exception {
    final IntArrayTree expectedTree = new IntArrayTree(); //(0)->(1, (2)->((3)->(5, 6), (4)->(7, 8)))
    {
      expectedTree.addTo(0); //1 (leaf)
      expectedTree.addTo(0); //2 (intern)
      expectedTree.addTo(2); //3 (intern
      expectedTree.addTo(2); //4 (intern)
      expectedTree.addTo(3); //5 (leaf)
      expectedTree.addTo(3); //6 (leaf)
      expectedTree.addTo(4); //7 (leaf)
      expectedTree.addTo(4); //8 (leaf)
    }

    final IntSeq target = MCTools.transformRegressionToMC(learn.target(L2.class).target, 16, new TDoubleArrayList());
    final HierTools.TreeBuilder builder = new HierTools.TreeBuilder(450);
    builder.createFromOrderedMulticlass(target);

    final IntTree actualTree = builder.releaseTree();
    final TIntIntMap actualMapping = builder.releaseMapping();

    assertEquals(expectedTree, actualTree);
    assertTrue(actualMapping.get(0) == 1);
    assertTrue(actualMapping.get(1) == 5);
    assertTrue(actualMapping.get(2) == 6);
    assertTrue(actualMapping.get(3) == 7);
    for (int i = 4; i < 16; i++) {
      assertTrue(actualMapping.get(i) == 8);
    }
  }
}
