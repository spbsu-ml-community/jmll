package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface Func {
  int xdim();
  @Nullable
  VecFunc gradient();
  double value(Vec x);
  Vec value(Mx x);
}
