package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.random.FastRandom;

public class XavierFiller implements Filler {
  private final int nIn;
  private final int nOut;
  private static final FastRandom rng = new FastRandom();

  public XavierFiller(int nIn, int nOut) {
    this.nIn = nIn;
    this.nOut = nOut;
  }

  @Override
  public void apply(Vec weights) {
    final double scale = 2. / (nIn + nOut);
    VecTools.fillUniform(weights, rng, scale);
  }
}
