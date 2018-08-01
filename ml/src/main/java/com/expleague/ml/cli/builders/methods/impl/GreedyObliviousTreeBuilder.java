package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.ml.BFGrid;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.GreedyObliviousTree;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GreedyObliviousTreeBuilder implements Factory<VecOptimization> {
  public static Factory<BFGrid> defaultGridBuilder;

  private Factory<BFGrid> gridBuilder = defaultGridBuilder;
  private int depth = 6;

  public void setDepth(final int d) {
    this.depth = d;
  }

  public void setGridBuilder(final Factory<BFGrid> gridBuilder) {
    this.gridBuilder = gridBuilder;
  }

  @Override
  public VecOptimization create() {
    return new GreedyObliviousTree(gridBuilder.create(), depth);
  }
}
