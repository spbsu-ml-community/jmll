package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.RandomnessAwareGradientBoosting;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.BinOptimizedRandomnessPolicy;
import com.expleague.ml.models.RandomVariableRandomnessPolicy;
import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;

/**
 * User: noxoomo
 */
public class RandomnessAwareGradientBoostingBuilder implements Factory<RandomnessAwareVecOptimization> {
  private final Factory<RandomnessAwareVecOptimization> defaultWeakBuilder = new GreedyRandomnessAwareObliviousTreeBuilder();

  private RandomnessAwareVecOptimization weak;
  private String lossName = "L2Reg";
  private double step = 0.005;
  private int iters = 1000;
  private ProcessRandomnessPolicy policy = BinOptimizedRandomnessPolicy.PointEstimateBin;

  public void setStep(final double s) {
    this.step = s;
  }

  public void setIterations(final int icount) {
    this.iters = icount;
  }

  public void setLocal(final String lossName) {
    this.lossName = lossName;
  }

  public void setWeak(final RandomnessAwareVecOptimization weak) {
    this.weak = weak;
  }

  public RandomnessAwareGradientBoostingBuilder setPolicy(final String policy) {
    this.policy = getPolicy(policy);
    return this;
  }

  ProcessRandomnessPolicy getPolicy(final String policy) {
    try {
      return BinOptimizedRandomnessPolicy.valueOf(policy);
    } catch (IllegalArgumentException e) {
      return RandomVariableRandomnessPolicy.valueOf(policy);
    }
  }


  @Override
  public RandomnessAwareVecOptimization create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    RandomnessAwareGradientBoosting.Config config = new RandomnessAwareGradientBoosting.Config(iters, step);
    //noinspection unchecked
    return new RandomnessAwareGradientBoosting((RandomnessAwareVecOptimization<L2>) weak, DataTools.targetByName(lossName),  config);
  }
}
