package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.models.nn.NeuralSpider;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import static java.lang.Math.max;

public class ConvCalcer implements NeuralSpider.NodeCalcer {
  private final int layerStart;
  private final int weightStart;
  private final int prevLayerStart;

  private final int numInputChannels;
  private final int height;
  private final int width;

  private final int numOutChannels;

  private final int kSizeX;
  private final int kSizeY;

  private final int strideX;
  private final int strideY;

  private final int paddX;
  private final int paddY;

  private final int weightPerState;

  public ConvCalcer(int layerStart, int weightStart, int prevLayerStart,
                    int kSizeX, int kSizeY, int strideX, int strideY, int paddX, int paddY,
                    int width, int height, int numInputChannels, int numOutChannels) {
    this.layerStart = layerStart;
    this.weightStart = weightStart;
    this.prevLayerStart = prevLayerStart;

    this.height = height;
    this.width = width;

    this.kSizeY = kSizeY;
    this.kSizeX = kSizeX;

    this.strideX = strideX;
    this.strideY = strideY;

    this.numInputChannels = numInputChannels;
    this.numOutChannels = numOutChannels;
    this.paddX = paddX;
    this.paddY = paddY;

    weightPerState = kSizeX * kSizeY * numInputChannels;
  }

  @Override
  public double apply(Vec state, Vec betta, int nodeIdx) {
    final int localIdx = nodeIdx - layerStart;
    final int localWIdx = localIdx % numOutChannels;
    final int wStart = weightStart + localWIdx * weightPerState;

    final int c_out = localIdx % width;
    final int y_out = ((localIdx - c_out) / width) % height;
    final int x_out = (((localIdx - c_out) / width) - y_out) / height;
    final int y = y_out * strideY;
    final int x = x_out * strideX;

    // TODO: ain't no padding now

    double result = 0.;
    for (int i = 0; i < kSizeX; i++) {
      for (int j = 0; j < kSizeY; j++) {
        for (int k = 0; k < numInputChannels; k++) {
          final int idx = prevLayerStart + ((x + i) * width + (y + j)) * numOutChannels + k;
          result += state.get(idx) * betta.get(wStart + i * kSizeY + j);
        }
      }
    }

    return max(result, 0.);
  }

  private int getX(int nodeIdx) {
    final int localIdx = nodeIdx - layerStart;
    final int c_out = localIdx % width;
    final int y_out = ((localIdx - c_out) / width) % height;
    final int x_out = (((localIdx - c_out) / width) - y_out) / height;
    return x_out * strideX;
  }

  @Override
  public int start(int nodeIdx) {
    return prevLayerStart + getX(nodeIdx) * width * numInputChannels;
  }

  @Override
  public int end(int nodeIdx) {
    final int endX = getX(nodeIdx) + kSizeX + 1;
    return prevLayerStart + endX * width * numInputChannels;
  }

  @Override
  public void gradByStateTo(Vec state, Vec betta, Vec to) {
    throw new NotImplementedException();
  }

  @Override
  public void gradByParametersTo(Vec state, Vec betta, Vec to) {
    throw new NotImplementedException();
  }
}
