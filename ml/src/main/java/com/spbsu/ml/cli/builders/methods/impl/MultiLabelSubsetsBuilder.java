package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multilabel.MultiLabelSubsetsMulticlass;

/**
 * User: qdeee
 * Date: 23.03.15
 */
public class MultiLabelSubsetsBuilder implements Factory<MultiLabelSubsetsMulticlass> {
  private int minExamplesCount = 50;
  private VecOptimization<BlockwiseMLLLogit> weakMulticlass;

  public void setMinCount(final int minExamplesCount) {
    this.minExamplesCount = minExamplesCount;
  }

  public void setWeak(final VecOptimization<BlockwiseMLLLogit> weakMulticlass) {
    this.weakMulticlass = weakMulticlass;
  }

  @Override
  public MultiLabelSubsetsMulticlass create() {
    return new MultiLabelSubsetsMulticlass(weakMulticlass, minExamplesCount);
  }
}
