package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multilabel.MultiLabelConflictMulticlass;

/**
 * User: qdeee
 * Date: 23.03.15
 */
public class MultiLabelConflictBuilder implements Factory<MultiLabelConflictMulticlass> {
  private VecOptimization<BlockwiseMLLLogit> weakMulticlass;
  private double threshold;
  private boolean zeroesEnabled;

  public void setWeak(final VecOptimization<BlockwiseMLLLogit> weakMulticlass) {
    this.weakMulticlass = weakMulticlass;
  }

  public void setThreshold(final double threshold) {
    this.threshold = threshold;
  }

  public void setZeroes(final boolean zeroesEnabled) {
    this.zeroesEnabled = zeroesEnabled;
  }

  @Override
  public MultiLabelConflictMulticlass create() {
    return new MultiLabelConflictMulticlass(weakMulticlass, threshold, zeroesEnabled);
  }
}
