package com.spbsu.ml.meta.impl;

import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.PoolFeatureMeta;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 17:32
 */
public class PoolFeatureMetaImpl implements PoolFeatureMeta {
  private final DataSet<?> owner;
  private final FeatureMeta delegate;

  public PoolFeatureMetaImpl(DataSet<?> owner, FeatureMeta delegate) {
    this.owner = owner;
    this.delegate = delegate;
  }

  @Override
  public DataSet<?> associated() {
    return owner;
  }

  @Override
  public String id() {
    return delegate.id();
  }

  @Override
  public String description() {
    return delegate.description() + " @" + owner.meta().id();
  }

  @Override
  public ValueType type() {
    return delegate.type();
  }
}
