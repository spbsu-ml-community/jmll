package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public interface Optimize<F extends Func> {
  Vec optimize(F func);
  Vec optimize(F func, Vec x0);
}
