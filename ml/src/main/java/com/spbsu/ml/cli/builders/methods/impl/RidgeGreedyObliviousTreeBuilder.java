package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.methods.trees.RidgeGreedyObliviousTree;

/**
 * User: noxoomo
 */

public class RidgeGreedyObliviousTreeBuilder implements Factory<VecOptimization> {
  private double lambda =2;
  private GreedyObliviousTree weak;
  private final GreedyObliviousTreeBuilder defaultWeakBuilder = new GreedyObliviousTreeBuilder();

  public void setWeak(final GreedyObliviousTree weak) {
    this.weak = weak;
  }

  public void setLambda(final double lambda) {
    this.lambda = lambda;
  }

  @Override
  public VecOptimization create() {
    if (weak == null) {
      weak = (GreedyObliviousTree) defaultWeakBuilder.create();

    }
    //noinspection unchecked
    return new RidgeGreedyObliviousTree(weak, lambda);
  }
}
