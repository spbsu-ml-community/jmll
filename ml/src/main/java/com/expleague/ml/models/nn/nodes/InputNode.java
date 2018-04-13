package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.vectors.Vec;

import static com.expleague.ml.models.nn.NeuralSpider.ForwardNode;

public class InputNode implements ForwardNode {
  @Override
  public double apply(Vec state, Vec betta, int nodeIdx) {
    return state.get(nodeIdx);
  }

  @Override
  public double activate(double value) {
    return value;
  }

  @Override
  public double grad(double value) {
    return 1.;
  }

  @Override
  public int start(int nodeIdx) {
    return 0;
  }

  @Override
  public int end(int nodeIdx) {
    return 0;
  }
}

