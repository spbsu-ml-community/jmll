package com.expleague.ml.models.nn.layers;

import com.expleague.ml.models.nn.NeuralSpider;
import com.expleague.ml.models.nn.nodes.ConvCalcer;

import java.util.Collections;
import java.util.List;

public class ConvLayer extends Layer3D {
  private final int height;
  private final int width;
  private final int numChannels;
  private final int paddX;
  private final int paddY;
  private final int weightLength;

  private final int kSizeX;
  private final int kSizeY;
  private final int strideX;
  private final int strideY;
  private final Layer3D prevLayer;

  public ConvLayer(int kSizeX, int kSizeY, int strideX, int strideY,
                   final Layer3D prevLayer, int numOutChannels, boolean padding) {
    paddX = padding ? (kSizeX - 1) / 2 : 0;
    paddY = padding ? (kSizeY - 1) / 2 : 0;

    height = (prevLayer.getHeight() + 2 * paddX - kSizeX) / strideX + 1;
    width = (prevLayer.getWidth() + 2 * paddY - kSizeY) / strideY + 1;
    numChannels = numOutChannels;
    weightLength = kSizeX * kSizeY * numOutChannels;
    this.kSizeX = kSizeX;
    this.kSizeY = kSizeY;
    this.strideX = strideX;
    this.strideY = strideY;
    this.prevLayer = prevLayer;
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
  public List<NeuralSpider.NodeCalcer> createCalcers(int prevLayerStart, int layerStart, int weightStart) {
    return Collections.nCopies(getStateLength(), new ConvCalcer(layerStart, weightStart, prevLayerStart,
        kSizeX, kSizeY, strideX, strideY, paddX, paddY,
        width, height, prevLayer.getNumChannels(), numChannels));
  }
}
