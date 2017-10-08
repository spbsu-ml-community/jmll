package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.func.generic.SubVecFuncC1;

public class MaxPool2DNode implements NeuralSpider.Node {
  private final int kHeight;
  private final int kWidth;
  private final InputView inputView;
  private final int rows;
  private final int cols;

  MaxPool2DNode(int kHeight, int kWidth, InputView inputView, int rows, int cols) {
    this.kHeight = kHeight;
    this.kWidth = kWidth;
    this.inputView = inputView;
    this.rows = rows;
    this.cols = cols;
  }

  class MaxPool extends FuncC1.Stub {
    private int maxIdx;

    @Override
    public double value(Vec state) {
      double value = state.get(0);
      maxIdx = 0;

      for (int i = 0; i < kHeight; i++) {
        for (int j = 0; j < kWidth; j++) {
          double current = state.get(i * cols + j);
          if (value < current) {
            value = current;
            maxIdx = i * cols + j;
          }
          value = (value > current) ? value : current;
        }
      }

      return value;
    }

    @Override
    public Vec gradientTo(Vec state, Vec to) {
      value(state);

      to.set(maxIdx, 1.);

      return to;
    }

    @Override
    public int dim() {
      return rows * cols;
    }
  }

  @Override
  public FuncC1 transByParameters(Vec betta) {
    return new SubVecFuncC1(
        new MaxPool(),
        inputView.stateStart, inputView.stateLength,
        inputView.stateStart + rows * cols);
  }

  @Override
  public FuncC1 transByParents(Vec state) {
    return new Const(new MaxPool().value(state));
  }
}
