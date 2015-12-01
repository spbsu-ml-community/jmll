package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.models.nn.NeuralSpider;

/**
* User: solar
* Date: 29.06.15
* Time: 17:27
*/
class NonDeterminedNode implements NeuralSpider.Node {
  private NFANetwork nfaNetwork;
  private final int lastLayerStart;
  private final NeuralSpider.Node[] nodes;

  public NonDeterminedNode(NFANetwork nfaNetwork, int lastLayerStart, NeuralSpider.Node[] nodes) {
    this.nfaNetwork = nfaNetwork;
    this.lastLayerStart = lastLayerStart;
    this.nodes = nodes;
  }

  @Override
  public FuncC1 transByParameters(final Vec betta) {
    return new FuncC1.Stub() {
      @Override
      public Vec gradientTo(Vec x, Vec to) {
        for (int i = 0; i < nfaNetwork.statesCount - 1; i++) {
          to.set(lastLayerStart + i, 1);
        }
        return to;
      }

      @Override
      public double value(Vec x) {
        double sum = 0;
        for (int i = 0; i < nfaNetwork.statesCount - 1; i++) {
          sum += x.get(lastLayerStart + i);
        }
        return sum;
      }

      @Override
      public int dim() {
        return nodes.length;
      }
    };
  }

  @Override
  public FuncC1 transByParents(Vec state) {
    double sum = 0;
    for (int i = 0; i < nfaNetwork.statesCount - 1; i++) {
      sum += state.get(lastLayerStart + i);
    }
    return new Const(sum);
  }
}
