package com.expleague.ml.data.perfectHash.impl;

import com.expleague.commons.func.Action;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.perfectHash.DynamicPerfectHash;
import com.expleague.ml.data.perfectHash.PerfectHash;

public class FeaturePerfectHash extends DynamicPerfectHash.Stub<Vec> implements DynamicPerfectHash<Vec> {
  private final int featureIndex;

  public FeaturePerfectHash(final int feature) {
    this.featureIndex = feature;
  }

  @Override
  protected double key(final Vec line) {
    return line.get(featureIndex);
  }

}


