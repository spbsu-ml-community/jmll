package com.expleague.ml;

import com.expleague.commons.func.Computable;
import com.expleague.ml.data.impl.BinarizedFeature;
import com.expleague.ml.data.impl.BinarizedFeatureExpectation;
import com.expleague.ml.data.impl.PointEstimateBinarizedFeature;
import com.expleague.ml.data.impl.SampledBinarizedFeature;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVec;
import com.expleague.ml.models.BinOptimizedRandomnessPolicy;
import com.expleague.ml.randomnessAware.DeterministicFeatureExtractor;
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
                                BinOptimizedRandomnessPolicy policy
  ) {
    if (!binarizedFeatures.containsKey(featureExtractor)) {
      final RandomVec feature = featureExtractor.computeAll(dataSet);
      final BinarizedFeature binarizedFeature;
      if (featureExtractor instanceof DeterministicFeatureExtractor) {
        binarizedFeature = new PointEstimateBinarizedFeature(featureBinarization, feature);
      }
      else {
        switch (policy) {
          case BinsExpectation: {
            binarizedFeature = new BinarizedFeatureExpectation(featureBinarization, feature);
            break;
          }
          case SampleBin: {
            binarizedFeature = new SampledBinarizedFeature(featureBinarization, feature);
            break;
          }
          case PointEstimateBin: {
            binarizedFeature = new PointEstimateBinarizedFeature(featureBinarization, feature);
            break;
          }
          default: {
            throw new RuntimeException("unknown policy");
          }
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
