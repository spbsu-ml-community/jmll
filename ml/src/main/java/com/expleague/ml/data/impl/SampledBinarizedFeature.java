package com.expleague.ml.data.impl;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.distributions.RandomVec;

import java.util.stream.IntStream;

public class SampledBinarizedFeature extends BinarizedFeature.Stub implements BinarizedFeature {

  public SampledBinarizedFeature(final FeatureBinarization binarization,
                                 final RandomVec feature) {
    super(binarization, feature);
  }

  @Override
  public final void visit(final int[] indices,
                          final FeatureVisitor visitor) {

    final FastRandom random = visitor.random();
    final int[] bins = new int[indices.length];

    IntStream.range(0, indices.length).parallel().forEach(i -> {
      final int idx = indices[i];
      bins[i] = binarization.bin(feature.instance(idx, random));
    });

    final double[] borders = binarization.borders();
    for (int i = 0; i < indices.length; ++i) {
      final int idx = indices[i];
      if (borders.length > 254) {
        visitor.accept(idx, i, bins[i], 1.0);
      } else {
        visitor.accept(idx, i, bins[i], 1.0);
      }
    }
  }
}
