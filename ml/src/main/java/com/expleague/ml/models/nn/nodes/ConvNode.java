package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.models.nn.NeuralSpider.BackwardNode;
import com.expleague.ml.models.nn.NeuralSpider.ForwardNode;
import com.expleague.ml.models.nn.layers.Layer;

public class ConvNode implements Layer.Node {
  private final int layerStart;
  private final int weightStart;
  private final int prevLayerStart;

  private final int numInputChannels;
  private final int prevWidth;
  private final int width;

  private final int numOutChannels;

  private final int kSizeX;
  private final int kSizeY;

  private final int strideX;
  private final int strideY;

  private final int paddX;
  private final int paddY;

  private final int weightPerState;
  private final int biasStart;

  private final AnalyticFunc activation;
  private final int height;
  private final int prevHeight;

  public ConvNode(int layerStart, int weightStart, int prevLayerStart,
                  int prevWidth, int prevHeight, int width, int height,
                   int kSizeX, int kSizeY, int strideX, int strideY, int paddX, int paddY,
                   int numInputChannels, int numOutChannels, AnalyticFunc activation) {
    this.layerStart = layerStart;
    this.weightStart = weightStart;
    this.prevLayerStart = prevLayerStart;

    this.prevWidth = prevWidth;
    this.prevHeight = prevHeight;
    this.width = width;
    this.height = height;

    this.kSizeY = kSizeY;
    this.kSizeX = kSizeX;

    this.strideX = strideX;
    this.strideY = strideY;

    this.numInputChannels = numInputChannels;
    this.numOutChannels = numOutChannels;
    this.paddX = paddX;
    this.paddY = paddY;

    this.activation = activation;

    weightPerState = kSizeX * kSizeY * numInputChannels;

    biasStart = weightStart + weightPerState * numOutChannels;
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
    private ForwardCalcer() {}

    @Override
    public double apply(Vec state, Vec betta, int nodeIdx) {
      final int localIdx = nodeIdx - layerStart;
      final int c_out = localIdx % numOutChannels;
      final int wStart = weightStart + c_out * weightPerState;

      final int y_out = (localIdx / numOutChannels) % width;
      final int x_out = localIdx / numOutChannels / width;
      final int y = y_out * strideY;
      final int x = x_out * strideX;

      // TODO: ain't no padding now

      double result = 0.;
      for (int i = 0; i < kSizeX; i++) {
        for (int j = 0; j < kSizeY; j++) {
          for (int k = 0; k < numInputChannels; k++) {
            final int idx = prevLayerStart + ((x + i) * prevWidth + (y + j)) * numInputChannels + k;
            final double a = state.get(idx);
            final double b = betta.get(wStart + (i * kSizeY + j) * numInputChannels + k);
            result += a * b;
          }
        }
      }
      result += betta.get(biasStart + c_out);

      return result;
    }

    @Override
    public double activate(double value) {
      return activation.value(value);
    }

    @Override
    public double grad(double value) {
      return activation.gradient(value);
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

      for (int c = 0; c < numOutChannels; c++) {
        final int wStart = weightStart + c * weightPerState;

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

            final int stateIdx = layerStart + (x_out * width + y_out) * numOutChannels + c;

            final double gradS = gradState.get(stateIdx);
            final double gradA = gradAct.get(stateIdx);

            result += betta.get(wStart + ((i - x) * kSizeY + (j - y)) * numInputChannels + k)
                * gradA * gradS;
          }
        }
      }

      return result;
    }

    @Override
    public int start(int nodeIdx) {
      final int localIdx = nodeIdx - prevLayerStart;
      final int i = localIdx / numInputChannels / prevWidth;
      final int x = Math.max((i - kSizeX) / strideX, 0);
      return layerStart + x * width * numOutChannels;
    }

    @Override
    public int end(int nodeIdx) {
      final int localIdx = nodeIdx - prevLayerStart;
      final int i = localIdx / numInputChannels / prevWidth;
      final int x = Math.min((i / strideX), height);
      return layerStart + x * width * numOutChannels;
    }
  }

  private class GradCalcer implements BackwardNode {
    @Override
    public double apply(Vec state, Vec gradState, Vec gradAct, Vec betta, int nodeIdx) {
      if (nodeIdx < biasStart) {
        final int localIdx = nodeIdx - weightStart;
        final int c = localIdx / weightPerState;
        final int i = (localIdx / numInputChannels / kSizeY) % kSizeX;
        final int j = (localIdx / numInputChannels) % kSizeY;
        final int k = localIdx % numInputChannels;

        double result = 0.;
        for (int x = 0; x < height; x++) {
          for (int y = 0; y < width; y++) {
            final int stateIdx = layerStart + (x * width + y) * numOutChannels + c;
            final int x_in = x * strideX + i;
            final int y_in = y * strideY + j;

            final double gradS = gradState.get(stateIdx);
            final double gradA = gradAct.get(stateIdx);
            final double s = state.get(prevLayerStart + (x_in * prevWidth + y_in) * numInputChannels + k);
            result += gradS * gradA * s;
          }
        }

        return result;
      }

      final int localIdx = nodeIdx - biasStart;
      final int c = localIdx % numOutChannels;
      double result = 0.;

      for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
          final int stateIdx = layerStart + (x * width + y) * numOutChannels + c;
          result += gradState.get(stateIdx) * gradAct.get(stateIdx);
        }
      }

      return result;
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
