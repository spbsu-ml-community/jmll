package com.spbsu.ml.meta.impl;

import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.meta.TargetMeta;

/**
 * User: qdeee
 * Date: 22.07.14
 */
public class FakeTargetMeta implements TargetMeta {
  private final DataSet<?> ds;
  private final ValueType valueType;

  public FakeTargetMeta(final DataSet<?> ds, final ValueType valueType) {
    this.ds = ds;
    this.valueType = valueType;
  }

  @Override
  public DataSet<?> associated() {
    return ds;
  }

  @Override
  public String id() {
    return "Fake id";
  }

  @Override
  public String description() {
    return "Fake description";
  }

  @Override
  public ValueType type() {
    return valueType;
  }
}
