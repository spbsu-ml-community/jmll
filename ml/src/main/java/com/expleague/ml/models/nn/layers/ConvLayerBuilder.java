package com.expleague.ml.models.nn.layers;

public class ConvLayerBuilder {
  private int kSizeX;
  private int kSizeY;
  private int strideX = 1;
  private int strideY = 1;
  private int width;
  private int height;
  private int numInputChannels;
  private int numOutChannels;
  private boolean padding = false;

  public void setHeight(int height) {
    this.height = height;
  }

  public void setkSize(int kSize) {
    this.kSizeX = kSize;
    this.kSizeY = kSizeY;
  }

  public void setkSizeX(int kSizeX) {
    this.kSizeX = kSizeX;
  }

  public void setkSizeY(int kSizeY) {
    this.kSizeY = kSizeY;
  }

  public void setStrideX(int strideX) {
    this.strideX = strideX;
  }

  public void setStrideY(int strideY) {
    this.strideY = strideY;
  }

  public void setWidth(int width) {
    this.width = width;
  }

  public void setNumInputChannels(int numInputChannels) {
    this.numInputChannels = numInputChannels;
  }

  public void setNumOutChannels(int numOutChannels) {
    this.numOutChannels = numOutChannels;
  }

  public void setPadding(boolean padding) {
    this.padding = padding;
  }

  public ConvLayer build() {
    return new ConvLayer(kSizeX, kSizeY, strideX, strideY, width, height,
        numInputChannels, numOutChannels, padding);
  }
}
