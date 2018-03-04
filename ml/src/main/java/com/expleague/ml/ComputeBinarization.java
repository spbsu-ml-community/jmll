package com.expleague.ml;

import com.expleague.commons.func.Computable;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.ctrs.Ctr;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;

import java.util.WeakHashMap;

/**
 * noxoomo
 */
public class ComputeBinarization implements Computable<VecDataSet, ComputeBinarization> {
  private WeakHashMap<VecRandomFeatureExtractor, FeatureBinarization> feature = new WeakHashMap<>();
  private VecDataSet dataSet;

  public synchronized FeatureBinarization computeBinarization(final VecRandomFeatureExtractor featureExtractor,
                                                              final FastRandom random,
                                                              final int binCount) {
    if (!feature.containsKey(featureExtractor)) {
      final FeatureBinarization.FeatureBinarizationBuilder featureBinarizationBuilder = new FeatureBinarization.FeatureBinarizationBuilder();
      if (featureExtractor instanceof Ctr) {
        featureBinarizationBuilder.setBinFactor(binCount / 2);
        featureBinarizationBuilder.useQuantileBinarization(true);
      } else {
        featureBinarizationBuilder.setBinFactor(binCount);
      }
      featureBinarizationBuilder.addSample(featureExtractor.computeAll(dataSet), random);
      if (OneHotFeaturesSet.isOneHot(featureExtractor)) {
        featureBinarizationBuilder.buildOneHot(true);
      }
      feature.put(featureExtractor, featureBinarizationBuilder.build(featureExtractor));
    }
    return feature.get(featureExtractor);
  }

  @Override
  public ComputeBinarization compute(final VecDataSet dataSet) {
    this.dataSet = dataSet;
    return this;
  }
}

