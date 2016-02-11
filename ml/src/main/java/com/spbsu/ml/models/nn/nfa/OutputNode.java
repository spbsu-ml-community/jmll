package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.models.nn.NeuralSpider;

/**
* User: solar
* Date: 29.06.15
* Time: 17:28
*/
class OutputNode implements NeuralSpider.Node {
  private final int statesCount;
  private final int finalStateIndex;
  private final NeuralSpider.Node[] nodes;

  public OutputNode(NeuralSpider.Node[] nodes, int statesCount, int finalStateIndex) {
    this.nodes = nodes;
    this.statesCount = statesCount;
    this.finalStateIndex = finalStateIndex;
  }

  @Override
  public FuncC1 transByParameters(final Vec betta) {
    return new FuncC1.Stub() {
      @Override
      public Vec gradientTo(Vec x, Vec to) {
        for (int i = statesCount + finalStateIndex; i < nodes.length; i += statesCount) {
          to.set(i, 1);
        }
        return to;
      }

      @Override
      public double value(Vec x) {
        double sum = 0;
        for (int i = statesCount + finalStateIndex; i < nodes.length; i += statesCount) {
          sum += x.get(i);
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
    for (int i = statesCount + finalStateIndex; i < nodes.length; i += statesCount) {
      sum += state.get(i);
    }
    return new Const(sum);
  }
}
