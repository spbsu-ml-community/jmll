package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;

public interface NetworkLayer {
  /**
   *
   * @param x input to the layer
   * @return array of output values of the nodes
   */
  Mx value(Mx x);

  /**
   *
   * @param x input to the layer
   * @param outputGrad gradient by output of this layer
   * @param isAfterValue set to true to optimize calculations. Can be used only after calling value on the same x
   * @return
   */
  LayerGrad gradient(Mx x, Mx outputGrad, boolean isAfterValue);

  void adjustParams(Vec dW);

  void setParams(Vec newW);

  int paramCount();

  Vec paramsView();

  class LayerGrad {
    public Vec gradByParams;
    public Mx gradByInput;

    LayerGrad(Vec gradByParams, Mx gradByInput) {
      this.gradByParams = gradByParams;
      this.gradByInput = gradByInput;
    }
  }
}
