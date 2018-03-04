package com.expleague.ml.data.impl;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.distributions.RandomVec;

import java.util.stream.IntStream;

public  class BinarizedFeatureExpectation extends BinarizedFeature.Stub implements BinarizedFeature {
  private final Mx weights;

  public BinarizedFeatureExpectation(final FeatureBinarization binarization,
                                     final RandomVec randomVec) {
    super(binarization, randomVec);
    final int docCount = randomVec.length();
    final double[] borders = binarization.borders();
    final int binCount = borders.length + 1;
    weights = new VecBasedMx(docCount, binCount);

    IntStream.range(0, docCount).parallel().forEach(i -> {
      double prevProb = 0.0;
      final double[] probs = new double[borders.length + 1];

      double totalProb = 0;
      for (int bin = 0; bin < borders.length; ++bin) {
        final double border = borders[bin];
        final double prob = feature.cdf(i, border);
        probs[bin] = prob - prevProb;
        probs[bin] = probs[bin] < 1e-5 ? 0 : probs[bin];
        totalProb += probs[bin];
        prevProb = prob;
      }
      probs[borders.length] = 1.0 - prevProb;
      probs[borders.length] = probs[borders.length] < 1e-5 ? 0 : probs[borders.length];
      totalProb += probs[borders.length];

      for (int bin = 0; bin <= borders.length; ++bin) {
        double p = probs[bin] / totalProb;
        weights.set(i, bin, p);
      }
    });
  }


  @Override
  public void visit(final int[] indices,
                    final FeatureVisitor visitor) {
    for (int i = 0; i < indices.length; ++i) {
      final int idx = indices[i];
      for (int bin = 0; bin < weights.columns(); ++bin) {
        if (weights.get(idx, bin) > 1e-9) {
          visitor.accept(idx, i, bin, weights.get(idx, bin));
        }
      }
    }
  }
}
