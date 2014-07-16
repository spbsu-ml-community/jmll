package com.spbsu.ml.meta.impl;

import com.spbsu.ml.meta.FeatureMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 13:44
 */
public class JsonFeatureMeta implements FeatureMeta {
  public String id;
  public String description;
  public ValueType type;

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
