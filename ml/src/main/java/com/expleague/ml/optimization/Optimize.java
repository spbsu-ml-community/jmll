package com.expleague.ml.optimization;

import com.expleague.commons.func.WeakListenerHolder;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.ProgressHandler;

import java.util.function.Function;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public interface Optimize<F extends Func> {
  Vec optimize(F func);
  Vec optimize(F func, Vec x0);

  default void projector(Function<Vec, Vec> projection) {
    throw new UnsupportedOperationException();
  }
}
