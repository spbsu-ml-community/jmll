package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multiclass.MultiClassOneVsRest;

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
