package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyRegion.cherry.GreedyTDCherryRegion;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GreedyTDCherryRegionBuilder implements Factory<VecOptimization> {
  public static Factory<BFGrid> defaultGridBuilder;

  private Factory<BFGrid> gridBuilder = defaultGridBuilder;

  public void setGridBuilder(final Factory<BFGrid> gridBuilder) {
    this.gridBuilder = gridBuilder;
  }

  @Override
  public VecOptimization create() {
    return new BootstrapOptimization( new GreedyTDCherryRegion(gridBuilder.create()), new FastRandom());
  }
}
