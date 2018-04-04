package com.expleague.ml.models.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.ArraySeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.SeqBuilder;
import com.expleague.ml.models.nn.NeuralSpider.NodeCalcer;
import com.expleague.ml.models.nn.layers.*;
import com.expleague.ml.models.nn.layers.InputLayerBuilder.InputLayer;
import com.expleague.ml.models.nn.layers.OutputLayerBuilder.OutputLayer;

import java.util.*;

public class NetworkBuilder<InType> {
  private List<LayerBuilder> layers = new ArrayList<>();
  private final List<List<Integer>> adjList = new ArrayList<>();
  private final Deque<LayerBuilder> topSort = new ArrayDeque<>();
  private InputLayerBuilder<InType> inputLayerBuilder;
  private OutputLayerBuilder outputLayerBuilder;

  private double dropout = 0.;

  public NetworkBuilder(InputLayerBuilder<InType> input) {
    inputLayerBuilder = input;
    layers.add(input);
    adjList.add(new ArrayList<>());
  }

  public NetworkBuilder<InType> setDropout(double dropout) {
    this.dropout = dropout;
    return this;
  }

  public NetworkBuilder<InType> append(LayerBuilder layer) {
    final int idx = layers.size();
    layers.add(layer);
    connectLayers(idx - 1, idx);
    return this;
  }

  public NetworkBuilder<InType> connect(LayerBuilder layerFrom, LayerBuilder layerTo) {
    final int idxFrom = layers.indexOf(layerFrom);
    int idxTo;
    if (layers.contains(layerTo)) {
      idxTo = layers.indexOf(layerTo);
    } else {
      idxTo = layers.size();
      layers.add(layerTo);
    }

    connectLayers(idxFrom, idxTo);
    return this;
  }

  private void connectLayers(int idxFrom, int idxTo) {
    layers.get(idxTo).setPrevBuilder(layers.get(idxFrom));
    adjList.get(idxFrom).add(idxTo);
    if (idxTo >= adjList.size()) {
      adjList.add(new ArrayList<>());
    }
  }

  public NetworkBuilder<InType> addSeq(LayerBuilder... layers) {
    for (final LayerBuilder layer: layers) {
      append(layer);
    }
    return this;
  }

  public InputLayerBuilder<InType> input() {
    return inputLayerBuilder;
  }

  public LayerBuilder last() {
    return layers.get(layers.size() - 1);
  }

  public Network build(OutputLayerBuilder output, LayerBuilder... outLayers) {
    this.outputLayerBuilder = output;

    final int size = layers.size();
    layers.add(output);
    if (outLayers.length == 0) {
      connectLayers(size - 1, size);
    }

    for (LayerBuilder layer: outLayers) {
      final int idxFrom = layers.indexOf(layer);
      if (idxFrom == -1) {
        throw new IllegalArgumentException("Layer chosen for output doesn't exist");
      }

      connectLayers(idxFrom, size);
    }

    buildGraph();

    return new Network();
  }

  private void buildGraph() {
    int numLayers = layers.size();
    boolean[] visited = new boolean[numLayers];
    Arrays.fill(visited, false);

    for (int i = 0; i < numLayers; i++) {
      if (!visited[i]) {
        visit(i, visited);
      }
    }
  }

  private void visit(int index, final boolean[] visited) {
    if (visited[index]) {
      return;
    }

    final List<Integer> list = adjList.get(index);
    for (int i = 0; i < list.size(); i++) {
      if (!visited[list.get(i)])
      visit(list.get(i), visited);
    }

    visited[index] = true;
    topSort.addFirst(layers.get(index));
  }

  public class Network {
    private InputLayer inputLayer;
    private OutputLayer outputLayer;
    private final List<Layer> layers = new ArrayList<>();
    private Seq<NodeCalcer> cacheCalcers;
    private int wdim;
    private int sdim;
    private int inputSize;

    private Network() {
      /* TODO: materialize here */
      if (inputLayerBuilder instanceof ConstSizeInput) {
        inputLayer = inputLayerBuilder.build();
        inputSize = inputLayer.xdim();
        cacheCalcers = materialize();
      }
    }

    InputLayerBuilder<InType> input() {
      return NetworkBuilder.this.input();
    }

    void initWeights(Vec weights) {
      for (Layer layer: layers) {
        layer.initWeights(weights);
      }
    }

    void setInput(InType input, Vec state) {
      inputLayerBuilder.setInput(input);
      inputLayer = inputLayerBuilder.build();
      inputLayer.toState(state);
    }

    Seq<NodeCalcer> materialize() {
      if (inputSize != inputLayer.xdim() || cacheCalcers == null) {
        int yStart = 0;
        int wStart = 0;
        SeqBuilder<NodeCalcer> seqBuilder = new ArraySeqBuilder<>(NodeCalcer.class);
        seqBuilder.addAll(inputLayer.materialize());

        Layer prevLayer = inputLayer;
        Iterator<LayerBuilder> layerIt = topSort.iterator();
        layerIt.next();
        while (layerIt.hasNext()) {
          LayerBuilder builder = layerIt.next();
          yStart += prevLayer.ydim();
          wStart += prevLayer.wdim();

          builder.yStart(yStart);
          builder.wStart(wStart);
          prevLayer = builder.build();
          seqBuilder.addAll(prevLayer.materialize());

          layers.add(prevLayer);
        }

        wdim = wStart;
        sdim = yStart;
        outputLayer = outputLayerBuilder.getLayer();
        return seqBuilder.build();
      }

      return cacheCalcers;
    }

    public int xdim(InType input) {
      return inputLayerBuilder.ydim(input);
    }

    public int stateDim() {
      return sdim;
    }

    public Vec outputFrom(Vec state) {
      return outputLayer.fromState(state);
    }

    public int wdim() {
      return wdim;
    }
  }
}
