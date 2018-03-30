package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;

public interface Filler {
  void apply(Vec weights);
}
