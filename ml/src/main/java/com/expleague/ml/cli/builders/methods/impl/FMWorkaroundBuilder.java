package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.ml.methods.FMTrainingWorkaround;
import com.expleague.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class FMWorkaroundBuilder implements Factory<VecOptimization> {
  private String task = "-r";
  private String dim = "1,1,8";
  private String iters = "1000";
  private String others = "";

  public void setTask(final String task) {
    this.task = task;
  }

  public void setDim(final String dim) {
    this.dim = dim;
  }

  public void setIters(final String iters) {
    this.iters = iters;
  }

  public void setOthers(final String others) {
    this.others = others;
  }

  @Override
  public VecOptimization create() {
    return new FMTrainingWorkaround(task, dim, iters, others);
  }
}
