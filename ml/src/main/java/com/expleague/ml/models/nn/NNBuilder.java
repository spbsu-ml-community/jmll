package com.expleague.ml.models.nn;

import com.expleague.ml.models.nn.layers.Layer;

import java.util.HashSet;
import java.util.List;

public class NNBuilder {
  private int inputDim;
  private int weightLength;
  private int outputDim;
  private final HashSet<Layer> seenLayers = new HashSet<>();
  private List<NeuralSpider.NodeCalcer> nodeCalcers;
  private double dropout = 0.;

  public NNBuilder setDropout(double dropout) {
    this.dropout = dropout;
    return this;
  }

  public ConvNet build(List<Layer> inputs, List<Layer> outputs) {
    seenLayers.addAll(outputs);

    for (Layer input: inputs) {
      inputDim += input.getStateLength();
      visit(input);
    }

    for (Layer output: outputs) {
      processLayer(output);
      outputDim += output.getStateLength();
    }

    return new ConvNet(nodeCalcers, inputDim, weightLength, outputDim, dropout);
  }

  private void processLayer(final Layer layer) {
    seenLayers.add(layer);
    layer.addCalcers(nodeCalcers);
    weightLength += layer.getWeightLength();
  }

  private void visit(final Layer layer) {
    if (seenLayers.contains(layer)) {
      return;
    }

    processLayer(layer);

    for (final Layer child: layer.getChildren()) {
      visit(child);
    }
  }
}
