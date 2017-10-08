package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.AnalyticFunc;

import java.util.*;

public class NetBuilder {
  private final List<NeuralSpider.Node> nodes = new ArrayList<>();
  private final LinkedHashMap<String, Layer> layerHashMap = new LinkedHashMap<>();
  public static final String DATA_LAYER_NAME = "data";
  private int weightSize;
  private int stateSize;
  private final ArrayList<Integer> layersSize = new ArrayList<>();
  private float dropout = 0.5f;
  private Random rng = new Random();

  public NetBuilder(int numInput) {
    final Layer layer = new Layer(DATA_LAYER_NAME, 0, 0, 1, numInput);
    stateSize = numInput + 1; // +1 from Spider class =(
    layerHashMap.put(layer.getName(), layer);
    layersSize.add(stateSize);
  }

  public NetBuilder(int numChannels, int numRows, int numCols) {
    final Layer layer = new Layer3D(DATA_LAYER_NAME, 0, 0,
        1, numChannels, numRows, numCols);
    stateSize = numChannels * numRows * numCols + 1; // +1 from Spider class =(
    layerHashMap.put(layer.getName(), layer);
    layersSize.add(stateSize);
  }

  public void setDropout(float dropout) {
    this.dropout = dropout;
  }

  public void setDropout(float dropout, Random rng) {
    this.dropout = dropout;
    this.rng = rng;
  }

  public static class Layer {
    private final String name;
    private final InputView inputView;

    public Layer(String name, int weightStart, int weightLength, int stateStart, int stateLength) {
      this.name = name;
      inputView = new InputView(weightStart, weightLength, stateStart, stateLength);
    }

    public InputView getView() {
      return inputView;
    }

    @Override
    public int hashCode() {
      return name.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof Layer) {
        return ((Layer) o).name.equals(name);
      }

      return false;
    }

    public String getName() {
      return name;
    }
  }

  public static class Layer3D extends Layer {
    public final int numChannels;
    public final int numRows;
    public final int numCols;

    public Layer3D(String name, int weightStart, int weightLength, int stateStart,
                   int numChannels, int numRows, int numCols) {
      super(name, weightStart, weightLength, stateStart, numChannels * numRows * numCols);

      this.numChannels = numChannels;
      this.numRows = numRows;
      this.numCols = numCols;
    }
  }

  private static int getOutConvDim(int inSize, int kSize, int stride) {
    return (inSize - kSize) / stride + 1;
  }

  public NetBuilder conv2D(int kHeight, int kWidth, int strideX, int strideY,
                           int nMaps, AnalyticFunc activation,
                           String name, String namePrevLayer) {
    if (layerHashMap.containsKey(name)) {
      throw new IllegalArgumentException("Illegal argument, layer with name " + name + " already exist.");
    }

    if (!layerHashMap.containsKey(namePrevLayer)) {
      throw new IllegalArgumentException("Illegal argument, layer with name " + namePrevLayer + " doesn't exist.");
    }

    final Layer3D prevLayer = (Layer3D) layerHashMap.get(namePrevLayer);
    final InputView prevLayerView = prevLayer.getView();

    final int inX = prevLayer.numRows;
    final int inY = prevLayer.numCols;
    final int inCh = prevLayer.numChannels;

    final int outX = getOutConvDim(inX, kHeight, strideX);
    final int outY = getOutConvDim(inY, kWidth, strideY);
    final int kernelSize = kHeight * kWidth;
    final Layer layer = new Layer3D(name, weightSize, inCh * kernelSize * nMaps,
        stateSize, inCh * nMaps, outX, outY);

    InputView layerView = layer.getView();
    weightSize += layerView.weightLength;
    stateSize += layerView.stateLength;

    layerHashMap.put(name, layer);
    layersSize.add(inCh * nMaps * outX * outY);

    for (int channel = 0; channel < inCh; channel++)
      for (int nMap = 0; nMap < nMaps; nMap++)
        for (int i = 0; i < inX - kHeight + 1; i += strideX) {
          for (int j = 0; j < inY - kWidth + 1; j += strideY) {
            int weightShift = channel == 0 ? nMap * kernelSize : channel * nMap * kernelSize;
            final InputView inputView = new InputView(
              layerView.weightStart + weightShift,
              kHeight * kWidth,
              prevLayerView.stateStart + inX * inY * channel + i * inY + j,
              kHeight * inY
            );
            nodes.add(new Conv2DNode(kHeight, kWidth, inputView, inX, inY, activation));
          }
        }

    return this;
  }

  public NetBuilder maxPool2D(int kHeight, int kWidth, int strideX, int strideY,
                              String name, String namePrevLayer) {
    if (layerHashMap.containsKey(name)) {
      throw new IllegalArgumentException("Illegal argument, layer with name " + name + " already exist.");
    }

    if (!layerHashMap.containsKey(namePrevLayer)) {
      throw new IllegalArgumentException("Illegal argument, layer with name " + namePrevLayer + " doesn't exist.");
    }

    final Layer3D prevLayer = (Layer3D) layerHashMap.get(namePrevLayer);
    final InputView prevLayerView = prevLayer.getView();

    final int inX = prevLayer.numRows;
    final int inY = prevLayer.numCols;
    final int inCh = prevLayer.numChannels;

    final int outX = getOutConvDim(inX, kHeight, strideX);
    final int outY = getOutConvDim(inY, kWidth, strideY);
    final Layer layer = new Layer3D(name, weightSize, 0,
        stateSize, inCh, outX, outY);

    InputView layerView = layer.getView();
    weightSize += layerView.weightLength;
    stateSize += layerView.stateLength;

    layerHashMap.put(name, layer);
    layersSize.add(inCh * outX * outY);

    for (int channel = 0; channel < inCh; channel++)
      for (int i = 0; i < inX - kHeight + 1; i += strideX) {
        for (int j = 0; j < inY - kWidth + 1; j += strideY) {
          final InputView inputView = new InputView(
              layerView.weightStart, 0,
              prevLayerView.stateStart + inX * inY * channel + i * inY + j,
              kHeight * inY
          );
          nodes.add(new MaxPool2DNode(kHeight, kWidth, inputView, inX, inY));
        }
      }

    return this;
  }

  public NetBuilder dense(int nOut, AnalyticFunc activation,
                          String name, String namePrevLayer) {
    if (layerHashMap.containsKey(name)) {
      throw new IllegalArgumentException("Illegal argument, layer with name " + name + " already exist.");
    }

    if (!layerHashMap.containsKey(namePrevLayer)) {
      throw new IllegalArgumentException("Illegal argument, layer with name " + namePrevLayer + " doesn't exist.");
    }

    final Layer prevLayer = layerHashMap.get(namePrevLayer);
    final InputView prevLayerView = prevLayer.getView();
    final int nIn = prevLayerView.stateLength;

    final Layer layer = new Layer(name, weightSize, nIn * nOut, stateSize, nOut);

    layerHashMap.put(name, layer);
    layersSize.add(nOut);

    InputView layerView = layer.getView();
    weightSize += layerView.weightLength;
    stateSize += layerView.stateLength;

    for (int i = 0; i < nOut; i++) {
      final InputView inputView = new InputView(
          layerView.weightStart + i * nIn,
          nIn,
          prevLayerView.stateStart,
          prevLayerView.stateLength);
      nodes.add(new LinearNode(inputView, activation));
    }

    return this;
  }

  public ConvNet build() {
    return new ConvNet(layersSize, layerHashMap, rng, dropout, nodes, weightSize);
  }
}
