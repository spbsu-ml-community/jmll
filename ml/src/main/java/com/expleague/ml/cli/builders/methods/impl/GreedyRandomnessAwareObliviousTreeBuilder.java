package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureExtractorsBuilder;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.GreedyRandomnessAwareObliviousTree;

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
  private boolean sampled = true;
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

  public void setSampled(final boolean sampled) {
    this.sampled = sampled;
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
    final GreedyRandomnessAwareObliviousTree weak = new GreedyRandomnessAwareObliviousTree(depth, featureExtractorsBuilder.build(), binarization, sampled, random);
    weak.useBootstrap(bootstrap);
    weak.forceSampledSplit(forceSampledSplit);

    return weak;
  }
}
