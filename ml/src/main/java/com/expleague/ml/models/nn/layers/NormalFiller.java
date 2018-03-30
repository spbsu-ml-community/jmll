package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.random.FastRandom;

public class NormalFiller implements Filler {
  private static final FastRandom rng = new FastRandom();

  @Override
  public void apply(Vec weights) {
    VecTools.fillGaussian(weights, rng);
  }
}
