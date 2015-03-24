package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class BootstrapOptimizationBuilder implements Factory<VecOptimization> {
  public static FastRandom defaultRandom;

  private final GreedyObliviousTreeBuilder defaultWeakBuilder = new GreedyObliviousTreeBuilder();

  private VecOptimization<WeightedLoss> weak;
  private FastRandom random = defaultRandom;

  public void setWeak(final VecOptimization<WeightedLoss> vecOptimization) {
    this.weak = vecOptimization;
  }

  public void setRandom(final FastRandom random) {
    this.random = random;
  }

  @Override
  public VecOptimization create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    return new BootstrapOptimization(weak, random);
  }
}
