package com.expleague.ml.data.impl;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.distributions.samplers.RandomVecSampler;

import java.util.stream.IntStream;

public class SampledBinarizedFeature extends BinarizedFeature.Stub implements BinarizedFeature {

  public SampledBinarizedFeature(final FeatureBinarization binarization,
                                 final RandomVec feature) {
    super(binarization, feature);
  }

  @Override
  public final void visit(final int[] indices,
                    final FeatureVisitor visitor) {
    final RandomVecSampler sampler = feature.sampler();

    final FastRandom random = visitor.random();
    final int[] bins = new int[indices.length];
    IntStream.range(0, indices.length).parallel().forEach(i -> {
      final int idx = indices[i];
      bins[i] = binarization.bin(sampler.instance(random, idx));
    });

    final double[] borders = binarization.borders();
    for (int i = 0; i < indices.length; ++i) {
      final int idx = indices[i];
      if (borders.length > 254) {
          visitor.accept(idx, i, bins[i], 1.0);
        }
        else {
          visitor.accept(idx, i, bins[i], 1.0);
        }
    }
  }
}
