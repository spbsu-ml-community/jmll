package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.ml.methods.GradientBoosting;
import com.expleague.commons.func.Factory;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GradientBoostingBuilder implements Factory<VecOptimization> {
  private final Factory<VecOptimization> defaultWeakBuilder = new BootstrapOptimizationBuilder();

  private VecOptimization weak;
  private String lossName = "LogL2";
  private double step = 0.005;
  private int iters = 200;

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

  @Override
  public VecOptimization create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    //noinspection unchecked
    return new GradientBoosting(weak, DataTools.targetByName(lossName), iters, step);
  }
}
