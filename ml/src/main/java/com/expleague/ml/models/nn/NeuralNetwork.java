package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;

public interface NeuralNetwork<In, Out> {
  Out apply(In input);
  Vec weights();
}
