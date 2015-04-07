package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multilabel.MultiLabelConflictMulticlass;

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
