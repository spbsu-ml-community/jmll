package com.expleague.ml.models.nn.layers;

public class ConvLayerBuilder {
  private int kSizeX;
  private int kSizeY;
  private int strideX = 1;
  private int strideY = 1;
  private int numOutChannels;
  private boolean padding = false;

  public ConvLayerBuilder kSize(int kSize) {
    this.kSizeX = kSize;
    this.kSizeY = kSizeY;
    return this;
  }

  public ConvLayerBuilder kSizeX(int kSizeX) {
    this.kSizeX = kSizeX;
    return this;
  }

  public ConvLayerBuilder kSizeY(int kSizeY) {
    this.kSizeY = kSizeY;
    return this;
  }

  public ConvLayerBuilder strideX(int strideX) {
    this.strideX = strideX;
    return this;
  }

  public ConvLayerBuilder strideY(int strideY) {
    this.strideY = strideY;
    return this;
  }

  public ConvLayerBuilder setNumOutChannels(int numOutChannels) {
    this.numOutChannels = numOutChannels;
    return this;
  }

  public ConvLayerBuilder includePadding(boolean padding) {
    this.padding = padding;
    return this;
  }

  public Layer3D build(Layer3D prevLayer) {
    final ConvLayer layer = new ConvLayer(kSizeX, kSizeY, strideX, strideY, prevLayer, numOutChannels, padding);
    prevLayer.appendChild(layer);
    return layer;
  }
}
