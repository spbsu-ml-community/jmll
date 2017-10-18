package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.wrappers.MultiLabelWrapper;

/**
 * User: qdeee
 * Date: 03.04.15
 */
public class MultiLabelWrapperBuilder implements Factory<VecOptimization> {
  private VecOptimization<TargetFunc> strong;

  public void setStrong(final VecOptimization<TargetFunc> strong) {
    this.strong = strong;
  }

  @Override
  public VecOptimization create() {
    return new MultiLabelWrapper<>(strong);
  }
}
