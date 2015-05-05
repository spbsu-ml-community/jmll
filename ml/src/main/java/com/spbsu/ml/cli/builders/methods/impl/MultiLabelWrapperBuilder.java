package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.wrappers.MultiLabelWrapper;

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
