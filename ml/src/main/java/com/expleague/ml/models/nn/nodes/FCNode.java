package com.expleague.ml.models.nn.nodes;

import com.expleague.commons.math.AnalyticFunc;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.models.nn.layers.Layer;

import static com.expleague.ml.models.nn.NeuralSpider.BackwardNode;
import static com.expleague.ml.models.nn.NeuralSpider.ForwardNode;

public class FCNode implements Layer.Node {
  private final int prevStateStart;
  private final int prevStateLength;
  private final int weightStart;
  private final int biasStart;
  private final int weightPerState;
  private final int layerStart;
  private final int layerLength;
  private final AnalyticFunc activation;

  public FCNode(int layerStart, int layerLength,
                int prevStateStart, int prevStateLength,
                int weightStart, int weightLength,
                AnalyticFunc activation) {
    this.layerStart = layerStart;
    this.layerLength = layerLength;

    this.prevStateStart = prevStateStart;
    this.prevStateLength = prevStateLength;
    this.weightStart = weightStart;

    final int mxWeightLength = weightLength - layerLength;
    this.weightPerState = mxWeightLength / layerLength;
    this.biasStart = weightStart + mxWeightLength;

    this.activation = activation;
  }

  @Override
  public ForwardNode forward() {
    return new ForwardUnit();
  }

  @Override
  public BackwardNode backward() {
    return new BackwardCalcer();
  }

  @Override
  public BackwardNode gradient() {
    return new GradCalcer();
  }

  private class ForwardUnit implements ForwardNode {
    @Override
    public double apply(Vec state, Vec betta, int nodeIdx) {
      final int localIdx = nodeIdx - layerStart;
      final int wStart = weightStart + localIdx * weightPerState;
      return VecTools.multiply(state.sub(prevStateStart, prevStateLength),
          betta.sub(wStart, prevStateLength)) + betta.get(biasStart + localIdx);
    }

    @Override
    public double activate(double value) {
      return activation.value(value);
    }

    @Override
    public double grad(double value) {
      return activation.gradient(value);
    }

    @Override
    public int start(int nodeIdx) {
      return prevStateStart;
    }

    @Override
    public int end(int nodeIdx) {
      return prevStateStart + prevStateLength;
    }
  }

  private class BackwardCalcer implements BackwardNode {
    /** It is assumed that state is already computed by forward pass. */

    @Override
    public double apply(Vec state, Vec gradState, Vec gradAct, Vec betta, int nodeIdx) {
      final int localIdx = nodeIdx - prevStateStart;
      double result = 0.;
      for (int i = 0; i < layerLength; i++) {
        result += gradState.get(layerStart + i)
            * gradAct.get(layerStart + i)
            * betta.get(weightStart + weightPerState * i + localIdx);
      }
      return result;
    }

    @Override
    public int start(int nodeIdx) {
      return layerStart;
    }

    @Override
    public int end(int nodeIdx) {
      return layerStart + layerLength;
    }
  }

  private class GradCalcer implements BackwardNode {
    @Override
    public double apply(Vec state, Vec gradState, Vec gradAct, Vec betta, int wIdx) {
      if (wIdx < biasStart) {
        final int localIdx = wIdx - weightStart;
        final int stateAfterIdx = localIdx / weightPerState;
        final int stateBeforeIdx = localIdx % weightPerState;

        final double gradS = gradState.get(layerStart + stateAfterIdx);
        final double gradA = gradAct.get(layerStart + stateAfterIdx);
        final double s = state.get(prevStateStart + stateBeforeIdx);
        return gradS * gradA * s;
      }

      final int localIdx = wIdx - biasStart;
      assert(localIdx < layerLength);
      final double dTds = gradState.get(layerStart + localIdx);
      final double dAct = gradAct.get(layerStart + localIdx);
      return dTds * dAct;
    }

    @Override
    public int start(int nodeIdx) {
      return -1;
    }

    @Override
    public int end(int nodeIdx) {
      return -1;
    }
  }
}
