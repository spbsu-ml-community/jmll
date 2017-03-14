package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public interface Optimize<F extends Func> {
  public Vec optimize(F func);
}
