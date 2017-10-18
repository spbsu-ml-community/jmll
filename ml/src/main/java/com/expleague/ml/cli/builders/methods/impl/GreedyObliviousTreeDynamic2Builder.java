package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import com.expleague.ml.dynamicGrid.trees.GreedyObliviousTreeDynamic2;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.VecOptimization;

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
