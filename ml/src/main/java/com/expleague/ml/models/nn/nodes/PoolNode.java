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
  private final int height;

  private final int kSizeX;
  private final int kSizeY;

  private final int strideX;
  private final int strideY;

  public PoolNode(int layerStart, int prevLayerStart, int numInputChannels,
                  int prevWidth, int width, int height,
                  int kSizeX, int kSizeY, int strideX, int strideY) {
    this.layerStart = layerStart;
    this.prevLayerStart = prevLayerStart;
    this.numInputChannels = numInputChannels;
    this.prevWidth = prevWidth;
    this.width = width;
    this.height = height;
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
    return new BackwardNode.Stub();
  }

  private class ForwardCalcer implements ForwardNode {
    private volatile Vec cachedState;

    @Override
    public double apply(Vec state, Vec betta, int nodeIdx) {
      if (cachedState == null) {
        cachedState = state;
      }

      final int localIdx = nodeIdx - layerStart;
      final int c_out = localIdx % numInputChannels;
      final int y_out = (localIdx / numInputChannels) % width;
      final int x_out = localIdx / numInputChannels / width;
      final int y = y_out * strideY;
      final int x = x_out * strideX;

      double result = Double.NEGATIVE_INFINITY;
      int bestIdx = 0;

      for (int i = 0; i < kSizeX; i++) {
        for (int j = 0; j < kSizeY; j++) {
          final int idx = prevLayerStart + ((x + i) * prevWidth + (y + j)) * numInputChannels + c_out;
          final double value = state.get(idx);
          if (value > result) {
            result = value;
            bestIdx = idx;
          }
        }
      }

      return bestIdx;
    }

    @Override
    public double activate(double value) {
      return cachedState.get((int) value);
    }

    @Override
    public double grad(double value) {
      return value;
    }

    private int getX(int nodeIdx) {
      final int localIdx = nodeIdx - layerStart;
      final int x_out = localIdx / numInputChannels / width;
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
      final int localIdx = nodeIdx - prevLayerStart;
      final int i = localIdx / numInputChannels / prevWidth;
      final int j = (localIdx / numInputChannels) % prevWidth;
      final int k = localIdx % numInputChannels;

      double result = 0.;

      final int minX = Math.max(((i - kSizeX) / strideX), 0);
      final int minY = Math.max(((j - kSizeY) / strideY), 0);
      final int maxX = Math.min((i / strideX), height);
      final int maxY = Math.min((j / strideY), width);

      if (minX >= height || minY >= width) {
        return 0.;
      }

      for (int x_out = minX; x_out <= maxX; x_out++) {
        for (int y_out = minY; y_out <= maxY; y_out++) {
          final int x = x_out * strideX;
          final int y = y_out * strideY;
          if (x < i - kSizeX + 1 || x > i || y < j - kSizeY + 1 || y > j) {
            continue;
          }

          if (x_out >= height || y_out >= width) {
            continue;
          }

          final int stateIdx = layerStart +  (x_out * width + y_out) * numInputChannels + k;
          final int prevStateIdx = prevLayerStart + (x * prevWidth + y) * numInputChannels + k;

          final int index = (int) gradAct.get(stateIdx);
          if (index == nodeIdx) {
            result += gradState.get(stateIdx);
          }
        }
      }

      return result;
    }

    private int getX(int nodeIdx) {
      final int localIdx = nodeIdx - prevLayerStart;
      final int i = localIdx / numInputChannels / prevWidth;
      return (i - kSizeX + 1) / strideX;
    }

    @Override
    public int start(int nodeIdx) {
      return layerStart + getX(nodeIdx) * width * numInputChannels;
    }

    @Override
    public int end(int nodeIdx) {
      final int endX = getX(nodeIdx) + kSizeX;
      return layerStart + endX * width * numInputChannels;
    }
  }
}
