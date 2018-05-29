package com.expleague.ml.func;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;

public interface RegularizerFunc extends FuncC1 {
  double lambda();
  Vec project(Vec x);

  abstract class Stub extends FuncC1.Stub implements RegularizerFunc {
    @Override
    public double lambda() {
      return 1;
    }
  }
}
