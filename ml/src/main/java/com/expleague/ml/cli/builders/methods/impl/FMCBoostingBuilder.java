package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multiclass.gradfac.FMCBoosting;

/**
 * User: qdeee
 * Date: 18.05.17
 */
public class FMCBoostingBuilder implements Factory<FMCBoosting> {
  private final Factory<VecOptimization> defaultWeakBuilder = new BootstrapOptimizationBuilder();

  public static FastRandom defaultRandom;

  private VecOptimization weak;
  private String lossName = "LogL2";
  private double step = 0.005;
  private int iters = 200;
  private FastRandom random = defaultRandom;
  private double gamma;

  public void setStep(final double s) {
    this.step = s;
  }

  public void setIterations(final int icount) {
    this.iters = icount;
  }

  public void setLocal(final String lossName) {
    this.lossName = lossName;
  }

  public void setWeak(final VecOptimization weak) {
    this.weak = weak;
  }

  public void setGamma(double gamma) {
    this.gamma = gamma;
  }

  @Override
  public FMCBoosting create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    //noinspection unchecked
    return new FMCBoosting(
            new StochasticALS(random, gamma),
            weak,
            (Class<? extends L2>) DataTools.targetByName(lossName),
            iters,
            step
    );
  }
}
