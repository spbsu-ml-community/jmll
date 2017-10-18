package com.expleague.ml.meta.impl;

import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.PoolFeatureMeta;
import com.expleague.ml.data.set.DataSet;

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
