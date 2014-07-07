package com.spbsu.ml.data;

import com.spbsu.commons.func.CacheHolder;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Vectorization;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.PoolMeta;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:22
 */
public interface VectorizedRealTargetDataSet<T> extends DataSet<Vec> {
  DataSet<T> original();
  Vectorization<T> vectorization();
  double target(T x);

  int xdim();

  Mx data();
  Vec target();
}
