package com.expleague.ml.models.nn;

import com.expleague.ml.models.nn.NeuralSpider.NodeCalcer;
import com.expleague.ml.models.nn.layers.Layer;

import java.util.*;

public class NNBuilder {
  private int inputDim;
  private int weightLength;
  private int outputDim;
  private final HashSet<Layer> seenLayers = new HashSet<>();
  private final Queue<Layer> seqLayers = new ArrayDeque<>();

  private List<NodeCalcer> nodeCalcers;
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
      outputDim += output.getStateLength();
    }

    return new ConvNet(nodeCalcers, inputDim, weightLength, outputDim, dropout);
  }

  private void visit(final Layer layer) {
    if (seenLayers.contains(layer)) {
      return;
    }

    seenLayers.add(layer);

    for (final Layer child: layer.getChildren()) {
      visit(child);
    }

    seqLayers.add(layer);
  }

  private void addCalcers(final Layer layer) {
//    layer.createCalcers();
  }
}
