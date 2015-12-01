package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.models.nn.NeuralSpider;

/**
* User: solar
* Date: 29.06.15
* Time: 17:28
*/
class OutputNode implements NeuralSpider.Node {
  private final int[] outputNodesConnections;
  private final NeuralSpider.Node[] nodes;

  public OutputNode(int[] outputNodesConnections, NeuralSpider.Node[] nodes) {
    this.outputNodesConnections = outputNodesConnections;
    this.nodes = nodes;
  }

  @Override
  public FuncC1 transByParameters(final Vec betta) {
    return new FuncC1.Stub() {
      @Override
      public Vec gradientTo(Vec x, Vec to) {
        for (int i = 0; i < outputNodesConnections.length; i++) {
          to.set(outputNodesConnections[i], 1);
        }
        return to;
      }

      @Override
      public double value(Vec x) {
        double sum = 0;
        for (int i = 0; i < outputNodesConnections.length; i++) {
          sum += x.get(outputNodesConnections[i]);
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
    for (int i = 0; i < outputNodesConnections.length; i++) {
      sum += state.get(outputNodesConnections[i]);
    }
    return new Const(sum);
  }
}
