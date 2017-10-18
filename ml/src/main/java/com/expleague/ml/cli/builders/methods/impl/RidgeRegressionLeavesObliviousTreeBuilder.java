package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.BFGrid;
import com.expleague.ml.methods.MultipleVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.methods.trees.GreedyObliviousTreeWithVecOptimizationLeaves;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.linearRegressionExperiments.MultipleRidgeRegression;

/**
 * User: noxoomo
 */
public class RidgeRegressionLeavesObliviousTreeBuilder implements Factory<VecOptimization> {
  public static FastRandom defaultRandom;
  public static Factory<BFGrid> defaultGridBuilder;
  private Factory<BFGrid> gridBuilder = defaultGridBuilder;
  public FastRandom random = defaultRandom;
  private int depth = 6;
  private double lambda = 1.0;

  public void setLambda(final double lambda) {
    this.lambda = lambda;
  }

  public void setDepth(final int depth) {
    this.depth = depth;
  }

  public void setGridBuilder(final Factory<BFGrid> gridBuilder) {
    this.gridBuilder = gridBuilder;
  }

  public void setRandom(final FastRandom random) {
    this.random = random;
  }

  @Override
  public VecOptimization create() {
    final GreedyObliviousTree<WeightedLoss> tree = new GreedyObliviousTree<>(gridBuilder.create(),depth);
    final MultipleVecOptimization<L2> leafLearner = new MultipleRidgeRegression(lambda);
    return new GreedyObliviousTreeWithVecOptimizationLeaves(tree,leafLearner,random);
  }
}
