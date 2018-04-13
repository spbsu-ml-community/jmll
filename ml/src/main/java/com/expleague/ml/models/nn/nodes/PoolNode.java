package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.models.nn.NeuralSpider.BackwardNode;
import com.expleague.ml.models.nn.NeuralSpider.ForwardNode;
import com.expleague.ml.models.nn.layers.Layer;

public class PoolNode implements Layer.Node {
  private final int layerStart;
  private final int prevLayerStart;

  private final int numInputChannels;
  private final int prevWidth;
  private final int width;

  private final int numOutChannels;

  private final int kSizeX;
  private final int kSizeY;

  private final int strideX;
  private final int strideY;

  public PoolNode(int layerStart, int prevLayerStart, int numInputChannels,
                  int prevWidth, int width, int numOutChannels,
                  int kSizeX, int kSizeY, int strideX, int strideY) {
    this.layerStart = layerStart;
    this.prevLayerStart = prevLayerStart;
    this.numInputChannels = numInputChannels;
    this.prevWidth = prevWidth;
    this.width = width;
    this.numOutChannels = numOutChannels;
    this.kSizeX = kSizeX;
    this.kSizeY = kSizeY;
    this.strideX = strideX;
    this.strideY = strideY;
  }

  @Override
  public ForwardNode forward() {
    return new ForwardCalcer();
  }

  @Override
  public BackwardNode backward() {
    return new BackwardCalcer();
  }

  @Override
  public BackwardNode gradient() {
    return new GradCalcer();
  }

  private class ForwardCalcer implements ForwardNode {
    @Override
    public double apply(Vec state, Vec betta, int nodeIdx) {
      final int localIdx = nodeIdx - layerStart;
      final int c_out = localIdx % numOutChannels;
      final int y_out = (localIdx / numOutChannels) % width;
      final int x_out = localIdx / numOutChannels / width;
      final int y = y_out * strideY;
      final int x = x_out * strideX;

      double result = 0.;
      for (int i = 0; i < kSizeX; i++) {
        for (int j = 0; j < kSizeY; j++) {
          final int idx = prevLayerStart + ((x + i) * prevWidth + (y + j)) * numInputChannels + c_out;
          result = Double.max(state.get(idx), result);
        }
      }

      return result;
    }

    @Override
    public double activate(double value) {
      return value;
    }

    @Override
    public double grad(double value) {
      return 1.;
    }

    private int getX(int nodeIdx) {
      final int localIdx = nodeIdx - layerStart;
      final int x_out = localIdx / numOutChannels / width;
      return x_out * strideX;
    }

    @Override
    public int start(int nodeIdx) {
      return prevLayerStart + getX(nodeIdx) * prevWidth * numInputChannels;
    }

    @Override
    public int end(int nodeIdx) {
      final int endX = getX(nodeIdx) + kSizeX;
      return prevLayerStart + endX * prevWidth * numInputChannels;
    }
  }

  private class BackwardCalcer implements BackwardNode {
    @Override
    public double apply(Vec state, Vec gradState, Vec gradAct, Vec betta, int nodeIdx) {
      return 0;
    }

    @Override
    public int start(int nodeIdx) {
      return 0;
    }

    @Override
    public int end(int nodeIdx) {
      return 0;
    }
  }

  private class GradCalcer implements BackwardNode {
    @Override
    public double apply(Vec state, Vec gradState, Vec gradAct, Vec betta, int nodeIdx) {
      return 0;
    }

    @Override
    public int start(int nodeIdx) {
      return 0;
    }

    @Override
    public int end(int nodeIdx) {
      return 0;
    }
  }
}
