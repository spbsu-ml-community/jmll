package com.spbsu.ml.loss.multiclass.hier;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.HierarchyTree;
import gnu.trove.iterator.TIntIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.Iterator;

/**
 * User: qdeee
 * Date: 06.03.14
 */
public abstract class HierLoss extends Func.Stub {
  protected HierarchyTree hierarchy;
  protected Vec target;
  protected TIntIntHashMap targetMapping;
  private int minEntries;

  protected HierLoss(HierarchyTree unfilledHierarchy, DataSet dataSet, int minEntries) {
    this.minEntries = minEntries;
    unfilledHierarchy.fill(dataSet);
    init(unfilledHierarchy, dataSet.target());
  }

  private void init(HierarchyTree filledHierarchy, Vec unmappedTarget) {
    HierarchyTree prunedTree = filledHierarchy.getPrunedCopy(minEntries);
    HierarchyTree.traversePrint(prunedTree.getRoot());

    final TIntIntHashMap newTargetMapping = HierarchyTree.getTargetMapping(filledHierarchy.getRoot(), prunedTree.getRoot());
    if (targetMapping != null) {
      for (TIntIntIterator iter = targetMapping.iterator(); iter.hasNext(); ) {
        iter.advance();
        final int oldVal = iter.value();
        final int newVal = newTargetMapping.get(oldVal);
        iter.setValue(newVal);

      }
      targetMapping.putAll(newTargetMapping);
    }
    else {
      this.targetMapping = newTargetMapping;
    }
    Vec newTarget = new ArrayVec(unmappedTarget.dim());
    for (int i = 0; i < newTarget.dim(); i++) {
      int oldTargetVal = (int) unmappedTarget.get(i);
      newTarget.set(i, targetMapping.get(oldTargetVal));
    }
    this.target = newTarget;
    this.hierarchy = prunedTree; //replace
  }

  protected HierLoss(HierLoss learningLoss, Vec testTarget) {
    this.hierarchy = learningLoss.hierarchy;
    this.targetMapping = learningLoss.targetMapping;

    Vec newTarget = new ArrayVec(testTarget.dim());
    for (int i = 0; i < newTarget.dim(); i++) {
      newTarget.set(i, targetMapping.get((int) testTarget.get(i)));
    }
    this.target = newTarget;
  }

  public HierarchyTree.Node getHierRoot() {
    return hierarchy.getRoot();
  }

  public void updateTree() {
    init(hierarchy, target);
  }

  @Override
  public int dim() {
    return target.dim();
  }
}