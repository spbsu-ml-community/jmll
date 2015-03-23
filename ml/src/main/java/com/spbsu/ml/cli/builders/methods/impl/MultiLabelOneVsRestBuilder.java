package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multilabel.MultiLabelOneVsRest;

/**
 * User: qdeee
 * Date: 23.03.15
 */
public class MultiLabelOneVsRestBuilder implements Factory<MultiLabelOneVsRest> {
  private VecOptimization<LLLogit> weakBinClass;

  public void setWeak(final VecOptimization<LLLogit> weakBinClass) {
    this.weakBinClass = weakBinClass;
  }

  @Override
  public MultiLabelOneVsRest create() {
    return new MultiLabelOneVsRest(weakBinClass);
  }
}
