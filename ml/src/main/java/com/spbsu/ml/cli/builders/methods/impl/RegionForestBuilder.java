package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyRegion.RegionForest;

/**
 * User: noxoomo
 * Date: 10.11.14
 */

public class RegionForestBuilder implements Factory<VecOptimization> {
  public static Factory<BFGrid> defaultGridBuilder;
  private double alpha = 0.02;
  private double beta = 0.5;
  private int maxFailed = 1;
  private int regionsCount = 5;
  private String meanMethod = "Naive";

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

  public void setRegionsCount(final int regionsCount) {
    this.regionsCount = regionsCount;
  }

  public void setMeanMethod(final String method) {
    this.meanMethod = method;
  }


  @Override
  public VecOptimization create() {
    return new RegionForest(gridBuilder.create(), new FastRandom(), regionsCount, RegionForest.MeanMethod.valueOf(meanMethod), alpha, beta, maxFailed);
  }
}
