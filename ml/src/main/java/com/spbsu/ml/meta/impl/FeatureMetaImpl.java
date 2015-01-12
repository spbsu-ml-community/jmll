package com.spbsu.ml.meta.impl;

import com.spbsu.ml.meta.FeatureMeta;

/**
 * User: solar
 * Date: 24.07.14
 * Time: 13:22
 */
public class FeatureMetaImpl extends FeatureMeta.Stub {
  private final ValueType type;
  private final String id;
  private final String description;

  public FeatureMetaImpl(final String id, final String description, final ValueType type) {
    this.type = type;
    this.id = id;
    this.description = description;
  }

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
}
