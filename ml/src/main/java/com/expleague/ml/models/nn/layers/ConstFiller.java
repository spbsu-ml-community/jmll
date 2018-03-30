package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

public class ConstFiller implements Filler {
  @Override
  public void apply(Vec weights) {
    VecTools.fill(weights, 0);
  }
}
