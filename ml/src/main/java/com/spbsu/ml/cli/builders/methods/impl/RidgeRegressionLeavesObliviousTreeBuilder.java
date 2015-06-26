package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.MultipleVecOptimization;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.linearRegressionExperiments.MultipleRidgeRegression;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.methods.trees.GreedyObliviousTreeWithVecOptimizationLeaves;

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
