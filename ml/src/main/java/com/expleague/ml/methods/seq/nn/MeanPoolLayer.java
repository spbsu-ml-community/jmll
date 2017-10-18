package com.expleague.ml.methods.seq.nn;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;

public class MeanPoolLayer implements NetworkLayer {
  @Override
  public Mx value(Mx x) {
    Mx result = new VecBasedMx(1, x.columns());
    for (int i = 0; i < x.rows(); i++) {
      for (int j = 0; j < x.columns(); j++) {
        result.adjust(0, j, x.get(i, j));
      }
    }
    VecTools.scale(result, 1.0 / x.rows());
    return result;
  }

  @Override
  public LayerGrad gradient(Mx x, Mx outputGrad, boolean isAfterValue) {
    Mx grad = new VecBasedMx(x.rows(), x.columns());
    for (int i = 0; i < x.rows(); i++) {
      for (int j = 0; j < x.columns(); j++) {
        grad.set(i, j, outputGrad.get(0, j) / x.rows());
      }
    }
    return new LayerGrad(new ArrayVec(), grad);
  }

  @Override
  public void adjustParams(Vec dW) {

  }

  @Override
  public void setParams(Vec newW) {

  }

  @Override
  public int paramCount() {
    return 0;
  }

  @Override
  public Vec paramsView() {
    return new ArrayVec();
  }
}
