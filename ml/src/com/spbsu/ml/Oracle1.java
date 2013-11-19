package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface Oracle1 extends Oracle0 {
  Vec gradient(Vec x);
  int dim();
}
