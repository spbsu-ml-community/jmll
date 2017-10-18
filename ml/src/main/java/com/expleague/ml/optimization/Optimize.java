package com.expleague.ml.optimization;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public interface Optimize<F extends Func> {
  Vec optimize(F func);
  Vec optimize(F func, Vec x0);
}
