package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.vectors.Vec;

public interface WeightInitializer {
  void apply(Vec weights);
}
