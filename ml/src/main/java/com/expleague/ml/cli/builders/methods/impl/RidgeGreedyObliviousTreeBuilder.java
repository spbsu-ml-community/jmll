package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.RidgeGreedyObliviousTree;
import com.expleague.ml.methods.trees.GreedyObliviousTree;

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
