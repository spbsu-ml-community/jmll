package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.func.generic.SubVecFuncC1;
import com.spbsu.ml.func.generic.Sum;
import com.spbsu.ml.models.nn.NeuralSpider;

/**
 * User: solar
 * Date: 29.06.15
 * Time: 17:27
 */
class NonDeterminedNode implements NeuralSpider.Node {
  private final int lastLayerStart;
  private final int notFinalStatesCount;
  private final int nodesCount;

  public NonDeterminedNode(int notFinalStatesCount, int lastLayerStart, int nodesCount) {
    this.lastLayerStart = lastLayerStart;
    this.notFinalStatesCount = notFinalStatesCount;
    this.nodesCount = nodesCount;
  }

  @Override
  public FuncC1 transByParameters(final Vec betta) {
    return new SubVecFuncC1(new Sum(), lastLayerStart, notFinalStatesCount, nodesCount);
  }

  @Override
  public FuncC1 transByParents(Vec state) {
    double sum = 0;
    for (int i = 0; i < notFinalStatesCount; i++) {
      sum += state.get(lastLayerStart + i);
    }
    return new Const(sum);
  }
}
