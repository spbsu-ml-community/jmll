package com.expleague.crawl.learning;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.crawl.learning.features.Feature;

import java.util.List;

/**
 * Created by noxoomo on 17/07/16.
 */
public class State {
  private final List<Feature> features;

  public State(final List<Feature> features) {
    this.features = features;
  }

  public List<Feature> features() {
    return features;
  }

  int dim() {
    int dim = 0;
    for (Feature feature : features) {
      dim += feature.dim();
    }
    return dim;
  }

  Vec vectorize() {
    final int dim = dim();
    final Vec vecFeatures = new ArrayVec(dim);
    int offset = 0;
    for (Feature feature : features) {
      for (int i = 0; i < feature.dim(); ++i) {
        vecFeatures.set(offset++, feature.at(i));
      }
    }
    return vecFeatures;
  }
}
