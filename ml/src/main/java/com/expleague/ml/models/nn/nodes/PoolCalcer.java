package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.models.nn.NeuralSpider;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class PoolCalcer implements NeuralSpider.NodeCalcer {
  private final int layerStart;
  private final int prevLayerStart;

  private final int numInputChannels;
  private final int height;
  private final int width;

  private final int kSizeX;
  private final int kSizeY;

  private final int strideX;
  private final int strideY;

  public PoolCalcer(int layerStart, int prevLayerStart,
                    int numInputChannels, int height, int width,
                    int kSizeX, int kSizeY, int strideX, int strideY) {
    this.layerStart = layerStart;
    this.prevLayerStart = prevLayerStart;
    this.numInputChannels = numInputChannels;
    this.height = height;
    this.width = width;
    this.kSizeX = kSizeX;
    this.kSizeY = kSizeY;
    this.strideX = strideX;
    this.strideY = strideY;
  }

  @Override
  public double apply(Vec state, Vec betta, int nodeIdx) {
    final int localIdx = nodeIdx - layerStart;

    final int c_out = localIdx % width;
    final int y_out = ((localIdx - c_out) / width) % height;
    final int x_out = (((localIdx - c_out) / width) - y_out) / height;
    final int y = y_out * strideY;
    final int x = x_out * strideX;

    // TODO: ain't no padding now

    double result = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < kSizeX; i++) {
      for (int j = 0; j < kSizeY; j++) {
        final int idx = prevLayerStart + ((x + i) * width + (y + j)) * numInputChannels + c_out;
        result = Double.max(state.get(idx), result);
      }
    }

    return result;
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
