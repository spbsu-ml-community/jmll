package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.VecOptimization;

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
