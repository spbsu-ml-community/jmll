package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureExtractorsBuilder;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.GreedyRandomnessAwareCtrTrans;
import com.expleague.ml.models.RandomVariableRandomnessPolicy;

import static com.expleague.ml.models.RandomVariableRandomnessPolicy.Expectation;

/**
 * User: noxoomo
 */
public class GreedyRandomnessAwareCtrTransBuilder implements Factory<VecOptimization> {
  public static FastRandom defaultRandom;
  public static FeatureExtractorsBuilder defaultFeaturesExtractorBuilder;
  private FeatureExtractorsBuilder featureExtractorsBuilder = defaultFeaturesExtractorBuilder;
  private FastRandom random = defaultRandom;
  private RandomVariableRandomnessPolicy randomnessPolicy = Expectation;

  public void setFeatureExtractorsBuilder(final FeatureExtractorsBuilder featureExtractorsBuilder) {
    this.featureExtractorsBuilder = featureExtractorsBuilder;
  }


  public void setPolicy(final String policy) {
    this.randomnessPolicy = RandomVariableRandomnessPolicy.valueOf(policy);
  }

  @Override
  public RandomnessAwareVecOptimization create() {
    final GreedyRandomnessAwareCtrTrans weak = new GreedyRandomnessAwareCtrTrans(featureExtractorsBuilder.hashes(), random);
    weak.setCtrEstimationPolicy(featureExtractorsBuilder.ctrEstimationPolicy());
    weak.setCtrEstimationOrder(featureExtractorsBuilder.ctrEstimationOrder());
    weak.setRandomnessPolicy(randomnessPolicy);
    return weak;
  }
}
