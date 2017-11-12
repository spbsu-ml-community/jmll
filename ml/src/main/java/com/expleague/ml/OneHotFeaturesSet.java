package com.expleague.ml;

import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;

import java.util.WeakHashMap;

public class OneHotFeaturesSet {
  private static WeakHashMap<VecRandomFeatureExtractor, Integer> oneHotFeatures = new WeakHashMap<>();

  public static synchronized void add(VecRandomFeatureExtractor extractor) {
    oneHotFeatures.put(extractor, 0);
  }

  public static boolean isOneHot(VecRandomFeatureExtractor extractor) {
    return oneHotFeatures.containsKey(extractor);
  }
}
