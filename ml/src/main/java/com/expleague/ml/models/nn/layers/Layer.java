package com.expleague.ml.models.nn.layers;

import com.expleague.ml.models.nn.NeuralSpider.NodeCalcer;

import java.util.List;

public abstract class Layer {
  private List<Layer> childLayers;

  public abstract int getStateLength();
  public abstract int getWeightLength();
  public abstract List<NodeCalcer> createCalcers(int prevLayerStart, int layerStart, int weightStart);

  public void appendChild(Layer layer) {
    childLayers.add(layer);
  }

  public List<Layer> getChildren() {
    return childLayers;
  }
}
