package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.AnalyticFunc;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.Seq;

public class ConvNode implements NeuralSpider.Node {
  private final AnalyticFunc activation;
  private final int dim;
  private final int inputDim;
  private final int[] inputShapes;
  private int nDims;
  private final int startIdx;

  ConvNode(Seq<Integer> filterShape, Seq<Integer> filterShift,
             Seq<Integer> inputShape, AnalyticFunc activation) {
    this.activation = activation;

    int dim = 1;
    int inputDim = 1;
    for (int i = 0; i < filterShape.length(); i++) {
      dim *= filterShape.at(i);
      inputDim *= inputShape.at(i);
    }
    this.dim = dim;
    this.inputDim = inputDim;

    nDims = filterShape.length();
    int[] filterShapes = new int[nDims];
    filterShapes[0] = 1;
    for (int i = 1; i < nDims; i++) {
      filterShapes[i] = filterShape.at(nDims - i) * filterShapes[i - 1];
    }

    inputShapes = new int[nDims];
    inputShapes[0] = 1;
    for (int i = 1; i < nDims; i++) {
      inputShapes[i] = inputShape.at(nDims - i) * inputShapes[i - 1];
    }

    int startIdx = 0;
    for (int k = 0; k < nDims; k++) {
      startIdx += filterShift.at(k) * inputShapes[k];
    }
    this.startIdx = startIdx;
  }

//  private double convolution(final Vec weights, final Vec state) {
//    double weightedSum = 0.;
//
//    int idx = startIdx;
//
//    for (int i = 0; i < dim; i++) {
//      for (int k = 0; (k < nDims) && (i % inputShapes[k] == 0); k++) {
//        idx += inputShapes[k];
//      }
//
//      weightedSum += state.get(idx) * weights.get(i);
//    }
//
//    return weightedSum;
//  }

  interface SpatialVisitor {
    void accept(double state, double weight, int stateIdx, int weightIdx);
  }

  private void applySpatial(final Vec weights, final Vec state, SpatialVisitor visitor) {
    int idx = startIdx;

    for (int i = 0; i < dim; i++) {
      for (int k = 0; (k < nDims) && (i % inputShapes[k] == 0); k++) {
        idx += inputShapes[k];
      }

      visitor.accept(state.get(idx), weights.get(i), idx, i);
    }
  }

  private double convolution(final Vec weights, final Vec state) {
    final double[] weightedSum = {0.};
    applySpatial(weights, state, (s, w, sIdx, wIdx) -> weightedSum[0] += s * w);
    return weightedSum[0];
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
      return inputDim;
    }

    @Override
    public Vec gradientTo(Vec state, Vec to) {
      double dT_dAct = activation.gradient(value(state));
      applySpatial(weights, state, (s, w, sIdx, wIdx) -> to.set(sIdx, dT_dAct * w));

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
      return dim;
    }

    @Override
    public Vec gradientTo(Vec weights, Vec to) {
      double dT_dAct = activation.gradient(value(weights));
      applySpatial(weights, state, (s, w, sIdx, wIdx) -> to.set(wIdx, dT_dAct * s));

      return to;
    }
  }

  @Override
  public FuncC1 transByParameters(Vec betta) {
    return new Convolution(betta);
  }

  @Override
  public FuncC1 transByParents(Vec state) {
    return new BackConvolution(state);
  }
}
