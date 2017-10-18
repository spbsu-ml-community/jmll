package com.expleague.ml;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.meta.FeatureMeta;

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
