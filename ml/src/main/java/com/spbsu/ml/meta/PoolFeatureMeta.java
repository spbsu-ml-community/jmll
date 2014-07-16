package com.spbsu.ml.meta;

import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 20.06.14
 * Time: 15:13
 */
public interface PoolFeatureMeta extends FeatureMeta {
  DataSet<?> associated();
}
