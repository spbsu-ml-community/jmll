package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;

public class LSTMLayer implements NetworkLayer {
  private final LSTMNode[] nodes;
  private Vec[][] nodeInputs;

  public LSTMLayer(int nodeCount, int inputDim, FastRandom random) {
    nodes = new LSTMNode[nodeCount];
    for (int i = 0; i < nodeCount; i++) {
      nodes[i] = new LSTMNode(inputDim, random);
    }
  }

  /**
   *
   * @param x i-th row of x is a signal value at the moment i
   * @return output.get(i, j) is a signal value of j-th node at the moment i
   */
  @Override
  public Mx value(Mx x) {
    Mx result = new VecBasedMx(x.rows(), nodes.length);
    nodeInputs = new Vec[nodes.length][x.rows()];

    for (int node = 0; node < nodes.length; node++) {
      Vec lastNodeOutput = new ArrayVec(2);
      for (int moment = 0; moment < x.rows(); moment++) {
        nodeInputs[node][moment] = VecTools.concat(x.row(moment), lastNodeOutput);
        lastNodeOutput = nodes[node].value(nodeInputs[node][moment]);
        result.set(moment, node, lastNodeOutput.get(0));
      }
    }

    return result;
  }

  //Fixme: for now assuming that this is the first layer in the network
  @Override
  public LayerGrad gradient(Mx x, Mx outputGrad, boolean isAfterValue) {
    if (!isAfterValue) {
      value(x);
    }

    final int paramCount = nodes[0].params().dim();
    Vec gradByParams = new ArrayVec(nodes.length * paramCount);
    Mx gradByInput = new VecBasedMx(x.rows(), x.columns());

    for (int node = 0; node < nodes.length; node++) {
      Vec nodeOutputGrad = new ArrayVec(2);
      for (int moment = x.rows() - 1; moment >= 0; moment--) {
        nodeOutputGrad.adjust(0, outputGrad.get(moment, node));
        NetworkNode.NodeGrad grad = nodes[node].grad(nodeInputs[node][moment], nodeOutputGrad);
        nodeOutputGrad = grad.gradByInput;
        VecTools.incscale(gradByParams.sub(node * paramCount, paramCount), grad.gradByParams, 1);
      }
    }

    return new LayerGrad(gradByParams, gradByInput);
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
    return nodes.length * nodes[0].params().dim();
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
