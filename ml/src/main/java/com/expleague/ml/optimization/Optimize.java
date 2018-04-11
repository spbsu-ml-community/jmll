package com.expleague.ml.optimization;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.ReguralizerFunc;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public interface Optimize<F extends Func> {
  Vec optimize(F func, ReguralizerFunc reg, Vec x0);

  default Vec optimize(F func) {
    return optimize(func, new ArrayVec(func.dim()));
  }

  default Vec optimize(F func, Vec x0) {
    return optimize(func, new UniformReguralizer(func.xdim()), x0);
  }
}
