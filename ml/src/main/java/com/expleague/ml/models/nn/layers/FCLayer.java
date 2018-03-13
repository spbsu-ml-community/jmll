package com.expleague.ml.models.nn.layers;

import com.expleague.ml.models.nn.NeuralSpider;

import java.util.List;

public class FCLayer extends Layer {
  @Override
  public int getStateLength() {
    return 0;
  }

  @Override
  public int getWeightLength() {
    return 0;
  }

  @Override
  public void createCalcers(List<NeuralSpider.NodeCalcer> calcers) {

  }
}
