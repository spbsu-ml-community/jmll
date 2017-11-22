package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureExtractorsBuilder;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.GreedyRandomnessAwareObliviousTree;
import com.expleague.ml.methods.trees.GreedyRandomnessAwareRegion;
import com.expleague.ml.models.BinOptimizedRandomnessPolicy;

/**
 * User: noxoomo
 */
public class GreedyRandomnessAwareRegionBuilder implements Factory<VecOptimization> {
  public static FastRandom defaultRandom;
  public static FeatureExtractorsBuilder defaultFeaturesExtractorBuilder;
  private FastRandom random = defaultRandom;
  private FeatureExtractorsBuilder featureExtractorsBuilder = defaultFeaturesExtractorBuilder;
  private int depth = 20;
  private int binarization = 32;
  private BinOptimizedRandomnessPolicy policy = BinOptimizedRandomnessPolicy.PointEstimateBin;
  private boolean bootstrap = false;
  private boolean forceSampledSplit = false;


  public GreedyRandomnessAwareRegionBuilder setBinarization(final int binarization) {
    this.binarization = binarization;
    return this;
  }

  public void setDepth(final int depth) {
    this.depth = depth;
  }

  public void setFeatureExtractorsBuilder(final FeatureExtractorsBuilder featureExtractorsBuilder) {
    this.featureExtractorsBuilder = featureExtractorsBuilder;
  }


  public GreedyRandomnessAwareRegionBuilder setPolicy(final String policy) {
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
    final GreedyRandomnessAwareRegion weak = new GreedyRandomnessAwareRegion(depth, featureExtractorsBuilder.build(), binarization, policy, random);
    weak.useBootstrap(bootstrap);
    weak.forceSampledSplit(forceSampledSplit);
    return weak;
  }
}
