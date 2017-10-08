package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.AnalyticFunc;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.func.generic.SubVecFuncC1;

public class Conv2DNode implements NeuralSpider.Node {
  private final int kHeight;
  private final int kWidth;
  private final int rows;
  private final int cols;
  private final InputView inputView;
  private final AnalyticFunc activation;

  Conv2DNode(int kHeight, int kWidth, InputView inputView,
             int rows, int cols, AnalyticFunc activation) {
    this.kHeight = kHeight;
    this.kWidth = kWidth;
    this.inputView = inputView;
    this.rows = rows;
    this.cols = cols;
    this.activation = activation;
  }

  private double convolution(final Vec weights, final Vec state) {
    double weightedSum = 0.;
    int counter = 0;

    for (int i = 0; i < kHeight; i++) {
      for (int j = 0; j < kWidth; j++) {
        weightedSum += state.get(i * cols + j) * weights.get(counter);
        counter++;
      }
    }

    return weightedSum;
  }

  private class Convolution extends FuncC1.Stub {
    public final Vec weights;

    private Convolution(Vec weights) {
      this.weights = weights;
    }

    @Override
    public double value(Vec state) {
      return activation.value(convolution(weights, state));
    }

    @Override
    public int dim() {
      return rows * cols;
    }

    @Override
    public Vec gradientTo(Vec state, Vec to) {
      double dT_dAct = activation.gradient(value(state));

      int counter = 0;
      for (int i = 0; i < kHeight; i++) {
        for (int j = 0; j < kWidth; j++) {
          to.set(i * cols + j, dT_dAct * weights.get(counter));
          counter++;
        }
      }

      return to;
    }
  }

  private class BackConvolution extends FuncC1.Stub {
    private final Vec state;

    BackConvolution(Vec state) {
      this.state = state;
    }

    @Override
    public double value(Vec weights) {
      return activation.value(convolution(weights, state));
    }

    @Override
    public int dim() {
      return kHeight * kWidth;
    }

    @Override
    public Vec gradientTo(Vec weights, Vec to) {
      double dT_dAct = activation.gradient(value(weights));

      int counter = 0;
      for (int i = 0; i < kHeight; i++) {
        for (int j = 0; j < kWidth; j++) {
          to.set(counter, dT_dAct * state.get(i * cols + j));
          counter++;
        }
      }

      return to;
    }
  }

  @Override
  public FuncC1 transByParameters(Vec betta) {
    return new SubVecFuncC1(
        new Convolution(betta.sub(inputView.weightStart, inputView.weightLength)),
        inputView.stateStart, inputView.stateLength,
        inputView.stateStart + rows * cols);
  }

  @Override
  public FuncC1 transByParents(Vec state) {
    return new SubVecFuncC1(
        new BackConvolution(state.sub(inputView.stateStart, inputView.stateLength)),
        inputView.weightStart, inputView.weightLength,
        inputView.weightStart + kHeight * kWidth);
  }
}
