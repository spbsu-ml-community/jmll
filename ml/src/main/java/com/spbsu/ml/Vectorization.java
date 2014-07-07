package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.meta.FeatureMeta;

/**
 * User: solar
 * Date: 20.06.14
 * Time: 15:07
 */
public interface Vectorization<T> {
  Vec value(T subject);
  FeatureMeta meta(int findex);
  int dim();
}
