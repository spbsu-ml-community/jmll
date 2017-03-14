package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyRegion.GreedyTDBumpyRegion;

/**
 * User: noxoomo
 */
public class GreedyTDBumpyRegionBuilder implements Factory<VecOptimization> {
  public static Factory<BFGrid> defaultGridBuilder;
  private double lambda = 0.02;

  private Factory<BFGrid> gridBuilder = defaultGridBuilder;

  public void setGridBuilder(final Factory<BFGrid> gridBuilder) {
    this.gridBuilder = gridBuilder;
  }

  public void setLambda(final double lambda) {
    this.lambda = lambda;
  }


  @Override
  public VecOptimization create() {
    return new GreedyTDBumpyRegion(gridBuilder.create(), lambda);
  }
}
