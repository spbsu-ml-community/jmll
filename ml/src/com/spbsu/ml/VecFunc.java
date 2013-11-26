package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.DataSet;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.11.13
 * Time: 10:26
 */
public interface VecFunc {
  int ydim();
  int xdim();
  Func direction(int direction);
  double value(Vec x, int direction);
  Vec value(DataSet ds, int classNo);
  Vec value(Vec x);
  Mx value(DataSet ds);
  @Nullable
  VecFunc gradient(int direction);
  Mx gradient(Vec x);

  Func[] directions();
}
