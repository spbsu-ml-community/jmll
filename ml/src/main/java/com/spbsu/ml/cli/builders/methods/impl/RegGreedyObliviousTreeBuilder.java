package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.methods.trees.RegGreedyObliviousTree;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class RegGreedyObliviousTreeBuilder implements Factory<VecOptimization> {
  public static Factory<BFGrid> defaultGridBuilder;

  private Factory<BFGrid> gridBuilder = defaultGridBuilder;
  private int depth = 6;
  private double lambda = 0.02;

  public void setDepth(final int d) {
    this.depth = d;
  }
  public void setLambda(double lambda) {
    this.lambda = lambda;
  }

  public void setGridBuilder(final Factory<BFGrid> gridBuilder) {
    this.gridBuilder = gridBuilder;
  }

  @Override
  public VecOptimization create() {
    return new RegGreedyObliviousTree(gridBuilder.create(), depth, lambda);
  }
}
