package com.expleague.ml.models.nn.nfa;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.func.generic.Const;
import com.expleague.ml.models.nn.NeuralSpider;

/**
* User: solar
* Date: 29.06.15
* Time: 17:27
*/
class InputNode implements NeuralSpider.Node {
  private final Const aConst;

  public InputNode(Const aConst) {
    this.aConst = aConst;
  }

  @Override
  public FuncC1 transByParameters(Vec betta) {
    return aConst;
  }

  @Override
  public FuncC1 transByParents(Vec state) {
    return aConst;
  }
}
