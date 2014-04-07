package com.spbsu.ml.loss.hier;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.Hierarchy;
import gnu.trove.list.TIntList;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 06.03.14
 */
public abstract class HierLoss extends Func.Stub {
  protected final Hierarchy hierarchy;
  protected final Vec target;
  protected final TIntIntHashMap targetMapping;
  protected final TIntList nodesClasses;

  protected HierLoss(Hierarchy unfilledHierarchy, DataSet dataSet, int minEntries) {
    unfilledHierarchy.fill(dataSet);
    Hierarchy prunedTree = unfilledHierarchy.getPrunedCopy(minEntries);
    Hierarchy.traversePrint(prunedTree.getRoot());

    this.targetMapping = Hierarchy.getTargetMapping(unfilledHierarchy.getRoot(), prunedTree.getRoot());
    Vec newTarget = new ArrayVec(dataSet.power());
    for (int i = 0; i < newTarget.dim(); i++) {
      int oldTargetVal = (int) dataSet.target().get(i);
      newTarget.set(i, targetMapping.get(oldTargetVal));
    }
    this.target = newTarget;
    this.hierarchy = prunedTree; //replace
    this.nodesClasses = Hierarchy.getPostorderedVisitOrder(prunedTree.getRoot(), false);
  }

  protected HierLoss(HierLoss learningLoss, Vec testTarget) {
    this.hierarchy = learningLoss.hierarchy;
    this.targetMapping = learningLoss.targetMapping;
    this.nodesClasses = learningLoss.nodesClasses;

    Vec newTarget = new ArrayVec(testTarget.dim());
    for (int i = 0; i < newTarget.dim(); i++) {
      newTarget.set(i, targetMapping.get((int) testTarget.get(i)));
    }
    this.target = newTarget;
  }

  public Hierarchy.CategoryNode getHierRoot() {
    return hierarchy.getRoot();
  }

  @Override
  public int dim() {
    return target.dim();
  }
}