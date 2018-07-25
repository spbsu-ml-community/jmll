package com.expleague.ml.meta;

import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.tools.Pool;

/**
 * User: solar
 * Date: 20.06.14
 * Time: 15:13
 */
public interface PoolFeatureMeta extends FeatureMeta {
  <T extends DSItem> Pool<T> owner();
  void setOwner(Pool<? extends DSItem> owner);

  default DataSet<?> associated() {
    return owner().data();
  }
}
