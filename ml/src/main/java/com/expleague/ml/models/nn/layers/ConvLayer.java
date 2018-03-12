package com.expleague.ml.models.nn.layers;

import com.expleague.ml.models.nn.NeuralSpider;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class ConvLayer extends Layer3D {
  private final int height;
  private final int width;
  private final int numChannels;
  private final int paddX;
  private final int paddY;
  private final int weightLength;

  public ConvLayer(int kSizeX, int kSizeY, int strideX, int strideY,
                   final Layer3D prevLayer, int numOutChannels, boolean padding) {
    paddX = padding ? (kSizeX - 1) / 2 : 0;
    paddY = padding ? (kSizeY - 1) / 2 : 0;

    height = (prevLayer.getHeight() + 2 * paddX - kSizeX) / strideX + 1;
    width = (prevLayer.getWidth() + 2 * paddY - kSizeY) / strideY + 1;
    numChannels = numOutChannels;
    weightLength = kSizeX * kSizeY * numOutChannels;
  }

  @Override
  public int getHeight() {
    return height;
  }

  @Override
  public int getWidth() {
    return width;
  }

  @Override
  public int getNumChannels() {
    return numChannels;
  }

  @Override
  public int getStateLength() {
    return width * height * numChannels;
  }

  @Override
  public int getWeightLength() {
    return weightLength;
  }

  @Override
  public void addCalcers(NeuralSpider.NodeCalcer[] calcers) {
    throw new NotImplementedException();
  }
}
