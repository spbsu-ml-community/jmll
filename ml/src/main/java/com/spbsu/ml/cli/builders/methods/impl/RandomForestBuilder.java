package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.RandomForest;
import com.spbsu.ml.methods.VecOptimization;

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
