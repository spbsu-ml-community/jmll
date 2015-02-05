package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyRegion.GreedyTDRegion;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GreedyTDRegionBuilder implements Factory<VecOptimization> {
  public static Factory<BFGrid> defaultGridBuilder;
  private double alpha = 0.02;
  private double beta = 0.5;
  private int maxFailed = 1;

  private Factory<BFGrid> gridBuilder = defaultGridBuilder;

  public void setGridBuilder(final Factory<BFGrid> gridBuilder) {
    this.gridBuilder = gridBuilder;
  }

  public void setAlpha(final double alpha) {
    this.alpha = alpha;
  }

  public void setBeta(final double beta) {
    this.beta = beta;
  }

  public void setMaxFailed(final int maxFailed) {
    this.maxFailed = maxFailed;
  }

  @Override
  public VecOptimization create() {
    return new BootstrapOptimization( new GreedyTDRegion(gridBuilder.create(), alpha, beta, maxFailed), new FastRandom());
  }
}
