package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.AnalyticFunc;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.func.generic.SubVecFuncC1;
import com.spbsu.ml.func.generic.WSum;

public class LinearNode implements NeuralSpider.Node {
  private final InputView inputView;
  private final AnalyticFunc activation;

  public LinearNode(InputView inputView, AnalyticFunc activation) {
    this.inputView = inputView;
    this.activation = activation;
  }

  private class Compose extends FuncC1.Stub {
    private final FuncC1 delegate;

    public Compose(Vec x) {
      delegate = new WSum(x);
    }

    @Override
    public double value(Vec x) {
      return activation.value(delegate.value(x));
    }

    @Override
    public Vec gradientTo(Vec x, Vec to) {
      double grad = activation.gradient(delegate.value(x));
      delegate.gradientTo(x, to);
      VecTools.scale(to, grad);

      return to;
    }

    @Override
    public int dim() {
      return inputView.stateLength;
    }
  }

  @Override
  public FuncC1 transByParameters(Vec betta) {
    return new SubVecFuncC1(
        new Compose(betta.sub(inputView.weightStart, inputView.weightLength)),
        inputView.stateStart, inputView.stateLength,
        inputView.stateStart + inputView.stateLength
    );
  }

  @Override
  public FuncC1 transByParents(Vec state) {
    return new SubVecFuncC1(
        new Compose(state.sub(inputView.stateStart, inputView.stateLength)),
        inputView.weightStart, inputView.weightLength,
        inputView.weightStart + inputView.weightLength
    );
  }
}
