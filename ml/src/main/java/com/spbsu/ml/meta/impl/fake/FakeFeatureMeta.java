package com.spbsu.ml.meta.impl.fake;

import com.spbsu.ml.meta.FeatureMeta;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 16:56
 */
public class FakeFeatureMeta implements FeatureMeta {
  protected final int id;
  private final ValueType type;

  public FakeFeatureMeta(int id, ValueType type) {
    this.id = id;
    this.type = type;
  }

  @Override
  public String id() {
    return "Fake feature #" + id;
  }

  @Override
  public String description() {
    return "Unknown source of feature";
  }

  @Override
  public ValueType type() {
    return type;
  }
}
