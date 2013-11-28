package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.11.13
 * Time: 10:26
 */
public interface VecFunc {
  int xdim();
  int ydim();
  Func direction(int direction);
  Func[] directions();

  double value(Vec x, int direction);
  Vec vvalue(Vec x);
  Mx value(Mx ds);

  @Nullable
  VecFunc gradient(int direction);
  Mx gradient(Vec x);
}
