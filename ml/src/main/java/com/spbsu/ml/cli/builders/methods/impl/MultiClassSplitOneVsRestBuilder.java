package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multiclass.MultiClassOneVsRest;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class MultiClassSplitOneVsRestBuilder implements Factory<MultiClassOneVsRest> {
  private VecOptimization learner;

  public void setLearner(final VecOptimization learner) {
    this.learner = learner;
  }

  @Override
  public MultiClassOneVsRest create() {
    if (learner == null) {
      throw new IllegalStateException("Learner was not set");
    }

    return new MultiClassOneVsRest(learner);
  }}
