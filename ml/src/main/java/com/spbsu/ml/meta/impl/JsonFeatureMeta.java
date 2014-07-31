package com.spbsu.ml.meta.impl;

import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.meta.*;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 13:44
 */
public class JsonFeatureMeta extends FeatureMeta.Stub implements PoolFeatureMeta {
  public String id;
  public String description;
  public ValueType type;
  public String associated;
  public Pool owner;

  @Override
  public String id() {
    return id;
  }

  @Override
  public String description() {
    return description;
  }

  @Override
  public ValueType type() {
    return type;
  }

  @Override
  public DataSet<?> associated() {
    return owner.data(associated);
  }
}
