package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;

public class LogisticLayer implements NetworkLayer {
  private final NetworkNode[] nodes;

  public LogisticLayer(int nodeCount, int inputDim, FastRandom random) {
    nodes = new NetworkNode[nodeCount];
    for (int i = 0; i < nodeCount; i++) {
      nodes[i] = new LogisticNode(inputDim, random);
    }
  }
  @Override
  public Mx value(Mx x) {
    Mx result = new VecBasedMx(x.rows(), nodes.length);
    for (int i = 0; i < x.rows(); i++) {
      for (int j = 0; j < nodes.length; j++) {
        result.set(i, j, nodes[j].value(x.row(i)).get(0));
      }
    }
    return result;
  }

  @Override
  public LayerGrad gradient(Mx x, Mx outputGrad, boolean isAfterValue) {
    final Vec gradByParam = new ArrayVec(paramCount());
    final Mx gradByInput = new VecBasedMx(x.rows(), x.columns());
    final int nodeParamCount = nodes[0].params().dim();
    for (int i = 0; i < x.rows(); i++) {
      for (int j = 0; j < nodes.length; j++) {
        NetworkNode.NodeGrad grad = nodes[j].grad(x.row(i), new SingleValueVec(outputGrad.get(i, j)));
        VecTools.append(gradByParam.sub(nodeParamCount * j, nodeParamCount), grad.gradByParams);
        VecTools.append(gradByInput.row(i), grad.gradByInput);
      }
    }
    return new LayerGrad(gradByParam, gradByInput);
  }

  @Override
  public void adjustParams(Vec dW) {
    final int nodeParamCount = nodes[0].params().dim();
    for (int i = 0; i < nodes.length; i++) {
      VecTools.append(nodes[i].params(), dW.sub(i * nodeParamCount, nodeParamCount));
    }
  }

  @Override
  public void setParams(Vec newW) {
    final int nodeParamCount = nodes[0].params().dim();
    for (int i = 0; i < nodes.length; i++) {
      VecTools.assign(nodes[i].params(), newW.sub(i * nodeParamCount, nodeParamCount));
    }
  }

  @Override
  public int paramCount() {
    return nodes[0].params().dim() * nodes.length;
  }

  @Override
  public Vec paramsView() {
    Vec[] params = new Vec[nodes.length];
    for (int i =0; i < nodes.length; i++) {
      params[i] = nodes[i].params();
    }
    return VecTools.join(params);
  }
}
