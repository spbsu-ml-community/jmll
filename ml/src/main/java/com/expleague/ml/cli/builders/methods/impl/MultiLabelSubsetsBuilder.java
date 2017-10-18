package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.multilabel.MultiLabelSubsetsMulticlass;
import com.expleague.commons.func.Factory;
import com.expleague.ml.methods.VecOptimization;

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
