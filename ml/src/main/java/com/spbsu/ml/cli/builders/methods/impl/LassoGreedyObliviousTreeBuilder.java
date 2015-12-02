package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.methods.trees.LassoGreedyObliviousTree;
import com.spbsu.ml.methods.trees.RidgeGreedyObliviousTree;

/**
 * User: noxoomo
 */

public class LassoGreedyObliviousTreeBuilder implements Factory<VecOptimization> {
  private double lambdaRatio = 1e-5;
  private GreedyObliviousTree weak;
  private final GreedyObliviousTreeBuilder defaultWeakBuilder = new GreedyObliviousTreeBuilder();

  public void setWeak(final GreedyObliviousTree weak) {
    this.weak = weak;
  }

  public void setLambdaRatio(final double lambdaRatio) {
    this.lambdaRatio = lambdaRatio;
  }

  @Override
  public VecOptimization create() {
    if (weak == null) {
      weak = (GreedyObliviousTree) defaultWeakBuilder.create();

    }
    //noinspection unchecked
    return new LassoGreedyObliviousTree(weak, lambdaRatio);
  }
}
