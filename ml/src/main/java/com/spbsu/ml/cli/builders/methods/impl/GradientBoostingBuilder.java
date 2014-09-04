package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GradientBoostingBuilder implements Factory<VecOptimization> {
  private Factory<VecOptimization> defaultWeakBuilder = new BootstrapOptimizationBuilder();

  private VecOptimization weak;
  private String lossName = "LogL2";
  private double step = 0.005;
  private int iters = 200;

  public void setStep(double s) {
    this.step = s;
  }

  public void setIterations(int icount) {
    this.iters = icount;
  }

  public void setLocal(String lossName) {
    this.lossName = lossName;
  }

  public void setWeak(VecOptimization weak) {
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
