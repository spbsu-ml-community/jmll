package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.trees.GreedyObliviousTreeDynamic2;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GreedyObliviousTreeDynamic2Builder implements Factory<VecOptimization> {
  public static Factory<DynamicGrid> defaultDynamicGridBuilder;

  private Factory<DynamicGrid> dynamicGridBuilder = defaultDynamicGridBuilder;
  private int depth = 6;
  private double lambda = 2;
  private int minSplits = 1;

  public void setDepth(final int depth) {
    this.depth = depth;
  }

  public void setLambda(final double lambda) {
    this.lambda = lambda;
  }

  public void setMinSplits(final int minSplits) {
    this.minSplits = minSplits;
  }

  public void setDynamicGridBuilder(final Factory<DynamicGrid> dynamicGridBuilder) {
    this.dynamicGridBuilder = dynamicGridBuilder;
  }

  @Override
  public VecOptimization create() {
    return new BootstrapOptimization(new GreedyObliviousTreeDynamic2(dynamicGridBuilder.create(), depth, lambda), new FastRandom());
  }}
