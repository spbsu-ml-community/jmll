package com.expleague.ml;

import com.expleague.commons.func.Computable;
import com.expleague.ml.data.impl.BinarizedFeature;
import com.expleague.ml.data.impl.BinarizedFeatureExpectation;
import com.expleague.ml.data.impl.PointEstimateBinarizedFeature;
import com.expleague.ml.data.impl.SampledBinarizedFeature;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.randomnessAware.DeterministicFeatureExctractor;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;

import java.util.WeakHashMap;

/**
 * noxoomo
 */
public class ComputeBinarizedFeature implements Computable<VecDataSet, ComputeBinarizedFeature> {
  private WeakHashMap<VecRandomFeatureExtractor, BinarizedFeature> binarizedFeatures = new WeakHashMap<>();
  private VecDataSet dataSet;

  public BinarizedFeature build(final VecRandomFeatureExtractor featureExtractor,
                                final FeatureBinarization featureBinarization,
                                boolean sampled
  ) {
    if (!binarizedFeatures.containsKey(featureExtractor)) {
      final RandomVec feature = featureExtractor.apply(dataSet);
      final BinarizedFeature binarizedFeature;
      if (featureExtractor instanceof DeterministicFeatureExctractor) {
        binarizedFeature = new PointEstimateBinarizedFeature(featureBinarization, feature);
      }
      else {
        if (sampled) {
          binarizedFeature = new SampledBinarizedFeature(featureBinarization, feature);
        }
        else {
          binarizedFeature = new BinarizedFeatureExpectation(featureBinarization, feature);
        }
      }
      synchronized (this) {
        binarizedFeatures.put(featureExtractor, binarizedFeature);
      }
    }

    return get(featureExtractor);
  }

  public synchronized BinarizedFeature get(final VecRandomFeatureExtractor featureExtractor) {
    if (binarizedFeatures.containsKey(featureExtractor)) {
      return binarizedFeatures.get(featureExtractor);
    }
    else {
      throw new RuntimeException("Unknown feature");
    }
  }

  @Override
  public ComputeBinarizedFeature compute(final VecDataSet dataSet) {
    this.dataSet = dataSet;
    return this;
  }
}
