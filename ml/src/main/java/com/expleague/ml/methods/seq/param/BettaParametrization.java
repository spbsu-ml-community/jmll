package com.expleague.ml.methods.seq.param;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;

public interface BettaParametrization {
  int paramCount(int stateCount);
  Mx getBettaMx(Vec params, int c, int stateCount);
  void gradientTo(Vec params, Mx dBetta, Vec dParam, int c, int stateCount);
}
