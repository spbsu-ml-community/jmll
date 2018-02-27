package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.models.nn.NeuralSpider;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class FCLayer implements NeuralSpider.NodeCalcer {
  private final int stateStart;
  private final int stateEnd;
  private final int prevStateStart;
  private final int prevStateLength;
  private final int weightStart;
  private final int weightPerState;

  public FCLayer(int stateStart, int stateLength,
                 int prevStateStart, int prevStateLength,
                 int weightStart, int weightLength) {
    this.stateStart = stateStart;
    this.prevStateStart = prevStateStart;
    this.prevStateLength = prevStateLength;
    this.weightStart = weightStart;
    this.weightPerState = weightLength / stateLength;
    this.stateEnd = stateStart + stateLength;
  }

  @Override
  public double apply(Vec state, Vec betta, int nodeIdx) {
    final int wStart = weightStart + (nodeIdx - stateStart) * weightPerState;
    double result = 0.;
    for (int i = 0; i < prevStateLength; i++) {
      result += state.get(i + prevStateStart) * betta.get(i + wStart);
    }
    return result;
  }

  @Override
  public int getStartNodeIdx() {
    return stateStart;
  }

  @Override
  public int getEndNodeIdx() {
    return stateEnd;
  }

  @Override
  public void gradByStateTo(Vec state, Vec betta, Vec to) {
    throw new NotImplementedException();
  }

  @Override
  public void gradByParametersTo(Vec state, Vec betta, Vec to) {
    throw new NotImplementedException();
  }
}
