package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.models.nn.NeuralSpider;

public class FCCalcer implements NeuralSpider.NodeCalcer {
  private final int prevStateStart;
  private final int prevStateLength;
  private final int weightStart;
  private final int weightPerState;
  private final int layerStart;
  private final AnalyticFunc activation;

  public FCCalcer(int layerStart, int stateLength,
                  int prevStateStart, int prevStateLength,
                  int weightStart, int weightLength,
                  AnalyticFunc activation) {
    this.layerStart = layerStart;
    this.prevStateStart = prevStateStart;
    this.prevStateLength = prevStateLength;
    this.weightStart = weightStart;
    this.weightPerState = weightLength / stateLength;
    this.activation = activation;
  }

  @Override
  public double apply(Vec state, Vec betta, int nodeIdx) {
    final int wStart = weightStart + (nodeIdx - layerStart) * weightPerState;
    final double result = VecTools.multiply(state.sub(prevStateStart, prevStateLength),
        betta.sub(wStart, prevStateLength));
    return activation.value(result);
  }

  @Override
  public int start(int nodeIdx) {
    return prevStateStart;
  }

  @Override
  public int end(int nodeIdx) {
    return prevStateStart + prevStateLength;
  }

  @Override
  public void gradByStateTo(Vec state, Vec betta, int nodeIdx, double wGrad, Vec gradState) { }

  @Override
  public void gradByParametersTo(Vec state, Vec betta, int nodeIdx, double sGrad, Vec gradW) { }
}
