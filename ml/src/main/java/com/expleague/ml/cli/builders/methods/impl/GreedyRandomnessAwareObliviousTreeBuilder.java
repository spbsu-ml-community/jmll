package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureExtractorsBuilder;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.GreedyRandomnessAwareCtrTrans;
import com.expleague.ml.methods.trees.GreedyRandomnessAwareObliviousTree;
import com.expleague.ml.models.BinOptimizedRandomnessPolicy;
import com.expleague.ml.models.RandomVariableRandomnessPolicy;

/**
 * User: noxoomo
 */
public class GreedyRandomnessAwareObliviousTreeBuilder implements Factory<VecOptimization> {
  public static FastRandom defaultRandom;
  public static FeatureExtractorsBuilder defaultFeaturesExtractorBuilder;
  private FastRandom random = defaultRandom;
  private FeatureExtractorsBuilder featureExtractorsBuilder = defaultFeaturesExtractorBuilder;
  private int depth = 6;
  private int binarization = 32;
  private BinOptimizedRandomnessPolicy policy = BinOptimizedRandomnessPolicy.PointEstimateBin;
  private boolean bootstrap = false;
  private boolean forceSampledSplit = false;


  public GreedyRandomnessAwareObliviousTreeBuilder setBinarization(final int binarization) {
    this.binarization = binarization;
    return this;
  }

  public void setDepth(final int depth) {
    this.depth = depth;
  }

  public void setFeatureExtractorsBuilder(final FeatureExtractorsBuilder featureExtractorsBuilder) {
    this.featureExtractorsBuilder = featureExtractorsBuilder;
  }

  public GreedyRandomnessAwareObliviousTreeBuilder setPolicy(final String policy) {
    this.policy = BinOptimizedRandomnessPolicy.valueOf(policy);
    return this;
  }

  public void setBootstrapped(final boolean bootstrap) {
    this.bootstrap = bootstrap;
  }

  public void setForceSampledSplit(final boolean flag) {
    this.forceSampledSplit = flag;
  }
  public void setRandom(final FastRandom random) {
    this.random = random;
  }

  @Override
  public RandomnessAwareVecOptimization create() {
    final GreedyRandomnessAwareObliviousTree weak = new GreedyRandomnessAwareObliviousTree(depth, featureExtractorsBuilder.build(), binarization, policy, random);
    weak.useBootstrap(bootstrap);
    weak.forceSampledSplit(forceSampledSplit);

    weak.setCtrEstimationPolicy(featureExtractorsBuilder.ctrEstimationPolicy());
    weak.setCtrEstimationOrder(featureExtractorsBuilder.ctrEstimationOrder());
    weak.setFeatureHashes(featureExtractorsBuilder.hashes());
    weak.setRandomnessPolicy(RandomVariableRandomnessPolicy.Expectation);
    return weak;
  }
}
