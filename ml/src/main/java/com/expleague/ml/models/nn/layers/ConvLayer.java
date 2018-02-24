package com.expleague.ml.models.nn.layers;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.models.nn.NeuralSpider;
import com.expleague.ml.models.nn.nodes.Conv2DNode;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class ConvLayer implements NeuralSpider.NodeType {
  private final int strideX;
  private final int strideY;
  private final int numOutChannels;
  private final int numInputChannels;
  private final boolean padding;
  private final int height;
  private final int width;
  private final int kSizeX;
  private final int kSizeY;

  public ConvLayer(int kSizeX, int kSizeY, int strideX, int strideY, int width, int height,
                   int numInputChannels, int numOutChannels, boolean padding) {
    this.strideX = strideX;
    this.strideY = strideY;
    this.numInputChannels = numInputChannels;
    this.numOutChannels = numOutChannels;
    this.padding = padding;
    this.height = height;
    this.width = width;
    this.kSizeX = kSizeX;
    this.kSizeY = kSizeY;
  }

  @Override
  public int getStateStart() {
    throw new NotImplementedException();
  }

  @Override
  public int getStateEnd() {
    throw new NotImplementedException();
  }

  @Override
  public Vec getState(Vec state, int nodeIdx) {
    throw new NotImplementedException();
  }

  @Override
  public Vec getWeight(Vec allWeights, int nodeIdx) {
    throw new NotImplementedException();
  }

  @Override
  public NeuralSpider.Node createNode() {
    return new Conv2DNode(this);
  }

  public int getStrideX() {
    return strideX;
  }

  public int getStrideY() {
    return strideY;
  }

  public int getNumOutChannels() {
    return numOutChannels;
  }

  public int getNumInputChannels() {
    return numInputChannels;
  }

  public boolean withPadding() {
    return padding;
  }

  public int getChannelSize() {
    return width * height;
  }

  public int getChannelWidth() {
    return width;
  }

  public int getKSizeX() {
    return kSizeX;
  }

  public int getKSizeY() {
    return kSizeY;
  }
}
