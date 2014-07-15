package com.spbsu.ml.loss.multiclass.hier;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.Func;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.impl.HierarchyTree;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.MLLLogit;
import gnu.trove.iterator.TIntIntIterator;
import gnu.trove.map.hash.TIntIntHashMap;

/**
 * User: qdeee
 * Date: 06.03.14
 */
public class HierLoss extends Func.Stub implements TargetFunc {
  @Override
  public double value(final Vec x) {
    return 0;
  }

  @Override
  public int dim() {
    return 0;
  }

  @Override
  public DataSet<?> owner() {
    return null;
  }
//  protected HierarchyTree hierarchy;
//  protected MLLLogit target;
//  public TIntIntHashMap targetMapping;
//  private int minEntries;
//
//  protected HierLoss(HierLoss learningLoss, MLLLogit testTarget) {
//    this.hierarchy = learningLoss.hierarchy;
//    this.targetMapping = learningLoss.targetMapping;
//
//    int[] newTarget = new int[testTarget.dim()];
//    for (int i = 0; i < newTarget.length; i++) {
//      int oldTargetVal = testTarget.label(i);
//      newTarget[i] = targetMapping.get(oldTargetVal);
//    }
//    this.target = new MLLLogit(new IntSeq(newTarget));
//  }
//
//  public MLLLogit original() {
//    return target;
//  }
//
//  protected HierLoss(HierarchyTree unfilledHierarchy, MLLLogit target, int minEntries) {
//    this.minEntries = minEntries;
//    unfilledHierarchy.fill(target);
//    init(unfilledHierarchy, target);
//  }
//
//  private void init(HierarchyTree filledHierarchy, MLLLogit target) {
//    HierarchyTree prunedTree = filledHierarchy.getPrunedCopy(minEntries);
//    HierarchyTree.traversePrint(prunedTree.getRoot());
//
//    final TIntIntHashMap newTargetMapping = HierarchyTree.getTargetMapping(filledHierarchy.getRoot(), prunedTree.getRoot());
//    if (targetMapping != null) {
//      for (TIntIntIterator iter = targetMapping.iterator(); iter.hasNext(); ) {
//        iter.advance();
//        final int oldVal = iter.value();
//        final int newVal = newTargetMapping.get(oldVal);
//        iter.setValue(newVal);
//
//      }
//      targetMapping.putAll(newTargetMapping);
//    }
//    else {
//      this.targetMapping = newTargetMapping;
//    }
//    int[] newTarget = new int[target.dim()];
//    for (int i = 0; i < newTarget.length; i++) {
//      int oldTargetVal = target.label(i);
//      newTarget[i] = targetMapping.get(oldTargetVal);
//    }
//    this.target = new MLLLogit(new IntSeq(newTarget));
//    this.hierarchy = prunedTree; //replace
//  }
//
//  public HierarchyTree.Node getHierRoot() {
//    return hierarchy.getRoot();
//  }
//
//  @Override
//  public int dim() {
//    return target.dim();
//  }
}