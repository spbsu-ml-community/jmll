package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.ml.dynamicGrid.trees.GreedyObliviousTreeDynamic;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.dynamicGrid.interfaces.DynamicGrid;
import com.expleague.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class GreedyObliviousTreeDynamicBuilder implements Factory<VecOptimization> {
  public static Factory<DynamicGrid> defaultDynamicGridBuilder;

  private Factory<DynamicGrid> dynamicGridBuilder = defaultDynamicGridBuilder;
  private int depth = 6;
  private double lambda = 2;

  public void setLambda(final double l) {
    this.lambda = l;
  }

  public void setDepth(final int d) {
    this.depth = d;
  }

  public void setDynamicGridBuilder(final Factory<DynamicGrid> dynamicGridBuilder) {
    this.dynamicGridBuilder = dynamicGridBuilder;
  }

  @Override
  public VecOptimization create() {
    return new BootstrapOptimization(new GreedyObliviousTreeDynamic(dynamicGridBuilder.create(), depth, lambda), new FastRandom());
  }
}
