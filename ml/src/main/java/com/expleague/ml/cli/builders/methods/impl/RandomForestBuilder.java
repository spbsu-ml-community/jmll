package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.methods.RandomForest;

/**
 * User: qdeee
 * Date: 27.01.15
 */
public class RandomForestBuilder implements Factory<RandomForest> {
  public static FastRandom defaultRandom;

  private final Factory<VecOptimization> defaultWeakBuilder = new GreedyObliviousTreeBuilder();
  private FastRandom random = defaultRandom;
  private int count;
  private VecOptimization<WeightedLoss> weak;

  public void setCount(final int count) {
    this.count = count;
  }

  public void setWeak(final VecOptimization weak) {
    this.weak = weak;
  }

  @Override
  public RandomForest create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }

    return new RandomForest(weak, random, count);
  }
}
