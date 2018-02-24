package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.models.nn.NeuralSpider;
import com.expleague.ml.models.nn.nodes.LinearNode;

public class FCLayer implements NeuralSpider.NodeType {
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
  public int getStateStart() {
    return stateStart;
  }

  @Override
  public int getStateEnd() {
    return stateEnd;
  }

  @Override
  public Vec getState(Vec state, int nodeIdx) {
    return state.sub(prevStateStart, prevStateLength);
  }

  @Override
  public Vec getWeight(Vec weights, int nodeIdx) {
    return weights.sub(weightStart +
        weightPerState * (nodeIdx - stateStart), prevStateLength);
  }

  @Override
  public NeuralSpider.Node createNode() {
    return new LinearNode();
  }
}
