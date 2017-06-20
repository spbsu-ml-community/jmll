package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.math.vectors.Vec;

public interface NetworkNode {
  Vec params();

  NodeGrad grad(Vec x, Vec nodeOutputGrad);

  Vec value(Vec input);

  class NodeGrad {
    public Vec gradByParams;
    public Vec gradByInput;

    NodeGrad(Vec gradByParams, Vec gradByInput) {
      this.gradByParams = gradByParams;
      this.gradByInput = gradByInput;
    }
  }
}
