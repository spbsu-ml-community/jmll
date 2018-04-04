package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.models.nn.NeuralSpider;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class FCCalcer implements NeuralSpider.NodeCalcer {
  private final int prevStateStart;
  private final int prevStateLength;
  private final int weightStart;
  private final int weightPerState;
  private final int layerStart;

  public FCCalcer(int layerStart, int stateLength,
                  int prevStateStart, int prevStateLength,
                  int weightStart, int weightLength) {
    this.layerStart = layerStart;
    this.prevStateStart = prevStateStart;
    this.prevStateLength = prevStateLength;
    this.weightStart = weightStart;
    this.weightPerState = weightLength / stateLength;
  }

  @Override
  public double apply(Vec state, Vec betta, int nodeIdx) {
    final int wStart = weightStart + (nodeIdx - layerStart) * weightPerState;
    final double result = VecTools.multiply(state.sub(prevStateStart, prevStateLength),
        betta.sub(wStart, prevStateLength));
    return result;
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
  public void gradByStateTo(Vec state, Vec betta, Vec to) {
    throw new NotImplementedException();
  }

  @Override
  public void gradByParametersTo(Vec state, Vec betta, Vec to) {
    throw new NotImplementedException();
  }
}
