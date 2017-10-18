package com.expleague.ml.methods.seq.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.func.generic.WSumSigmoid;

public class LogisticNode implements NetworkNode {
  private final Vec w;
  private final FuncC1 func;

  public LogisticNode(int inputDim, FastRandom random) {
    w = new ArrayVec(inputDim + 1);
    for (int i = 0; i < inputDim; i++) {
      w.set(i, random.nextGaussian() / inputDim);
    }
    func = new WSumSigmoid(w);
  }

  @Override
  public Vec params() {
    return w;
  }

  @Override
  public NodeGrad grad(Vec input, Vec nodeOutputGrad) {
    final double outputGrad = nodeOutputGrad.get(0);
    final Vec inputWithOne = VecTools.concat(input, new SingleValueVec(1));

    final Vec inputGrad = func.gradient(inputWithOne);
    VecTools.scale(inputGrad, outputGrad);

    final Vec paramsGrad  = new WSumSigmoid(inputWithOne).gradient(w);
    VecTools.scale(paramsGrad, outputGrad);

    return new NodeGrad(paramsGrad, inputGrad.sub(0, input.dim()));
  }

  @Override
  public Vec value(Vec input) {
    return new SingleValueVec(func.value(VecTools.concat(input, new SingleValueVec(1))));
  }
}
