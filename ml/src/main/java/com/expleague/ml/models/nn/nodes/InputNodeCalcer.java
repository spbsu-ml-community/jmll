package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.vectors.Vec;

import static com.expleague.ml.models.nn.NeuralSpider.NodeCalcer;

public class InputNodeCalcer implements NodeCalcer {
  @Override
  public double apply(Vec state, Vec betta, int nodeIdx) {
    return state.get(nodeIdx);
  }

  @Override
  public int start(int nodeIdx) {
    return 0;
  }

  @Override
  public int end(int nodeIdx) {
    return 0;
  }

  @Override
  public void gradByStateTo(Vec state, Vec betta, int nodeIdx, double wGrad, Vec gradState) { }

  @Override
  public void gradByParametersTo(Vec state, Vec betta, int nodeIdx, double sGrad, Vec gradW) { }
}

