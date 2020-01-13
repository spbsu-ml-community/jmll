package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.BFGrid;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.greedyRegion.GreedyTDCherryRegion;

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
