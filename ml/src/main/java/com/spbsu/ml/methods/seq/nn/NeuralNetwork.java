package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import java.util.List;

public class NeuralNetwork {
  private final NetworkLayer[] layers;
  private final int[] prefixParamCount;

  public NeuralNetwork(NetworkLayer ...layers) {
    this.layers = layers;

    prefixParamCount =  new int[layers.length + 1];
    for (int i = 0; i < layers.length; i++) {
      prefixParamCount[i + 1] = prefixParamCount[i] + layers[i].paramCount();
    }

  }

  public Mx value(Mx input) {
    Mx cur = input;
    for (NetworkLayer layer: layers) {
      cur = layer.value(cur);
    }
    return cur;
  }

  public Vec gradByParams(Mx input, Mx outputGrad, boolean isAfterValue) {
    Mx inputs[] = new Mx[layers.length + 1];
    inputs[0] = input;

    for (int i = 0; i < layers.length; i++) {
      inputs[i + 1] = layers[i].value(inputs[i]);
    }

    final Vec paramsGrad = new ArrayVec(prefixParamCount[layers.length]);


    for (int i = layers.length - 1; i >= 0; i--) {
      NetworkLayer.LayerGrad grad = layers[i].gradient(inputs[i], outputGrad, isAfterValue);
      VecTools.append(paramsGrad.sub(prefixParamCount[i], layers[i].paramCount()), grad.gradByParams);
      outputGrad = grad.gradByInput;
    }

    return paramsGrad;
  }

  public void adjustParams(Vec dW) {
    for (int i = 0; i < layers.length; i++) {
      layers[i].adjustParams(dW.sub(prefixParamCount[i], layers[i].paramCount()));
    }
  }

  public void setParams(Vec dW) {
    for (int i = 0; i < layers.length; i++) {
      layers[i].setParams(dW.sub(prefixParamCount[i], layers[i].paramCount()));
    }
  }

  public int paramCount() {
    return prefixParamCount[layers.length];
  }

  public Vec paramsView() {
    Vec[] params = new Vec[layers.length];
    for (int i = 0; i < layers.length; i++) {
      params[i] = layers[i].paramsView();
    }
    return VecTools.join(params);
  }
}
