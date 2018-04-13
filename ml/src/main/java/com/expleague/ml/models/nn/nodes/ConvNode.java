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

  public ConvNode(int layerStart, int weightStart, int prevLayerStart, int prevWidth, int width,
                   int kSizeX, int kSizeY, int strideX, int strideY, int paddX, int paddY,
                   int numInputChannels, int numOutChannels, AnalyticFunc activation) {
    this.layerStart = layerStart;
    this.weightStart = weightStart;
    this.prevLayerStart = prevLayerStart;

    this.prevWidth = prevWidth;
    this.width = width;

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
