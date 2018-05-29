package com.expleague.ml.methods.seq.param;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;

public interface WeightParametrization {
  Mx getMx(Vec params, int c, int stateCount);
  void gradientTo(Vec params, Vec out, Vec dOut, Vec dOutNew, Mx dW, int c, int stateCount);
}
