package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 20.06.14
 * Time: 15:07
 */
public interface Vectorization<T> {
  Vec value(T subject);
  int dim();
}
