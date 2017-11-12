package com.expleague.ml.data.impl;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.randomnessAware.DeterministicFeatureExctractor;

/**
 * User: noxoomo
 */


public interface BinarizedFeature {

  FeatureBinarization binarization();

  void visit(final int[] indices,
             final FeatureVisitor visitor);


  default boolean isDeterministic() {
    return binarization().owner() instanceof DeterministicFeatureExctractor;
  }

  //line is index in indices
  interface FeatureVisitor {

    void accept(final int idx, final int line, final int bin, final double prob);

    FastRandom random();

  }

  abstract class Stub implements BinarizedFeature {
    protected final FeatureBinarization binarization;
    protected final RandomVec feature;

    public Stub(final FeatureBinarization binarization, final RandomVec feature) {
      this.binarization = binarization;
      this.feature = feature;
    }

    public final FeatureBinarization binarization() {
      return binarization;
    }
  }
}
