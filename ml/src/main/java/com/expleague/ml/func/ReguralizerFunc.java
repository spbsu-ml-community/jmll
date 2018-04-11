package com.expleague.ml.func;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;

public interface ReguralizerFunc extends FuncC1 {
  double lambda();
  Vec project(Vec x);

  abstract class Stub extends FuncC1.Stub implements ReguralizerFunc {
    @Override
    public double lambda() {
      return 1;
    }
  }
}
