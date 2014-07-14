package com.spbsu.ml.meta.impl;

import com.spbsu.ml.meta.FeatureMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 13:44
 */
public class JsonLineMeta implements FeatureMeta {
  public String id;
  public String description;
  public Alignment alignment;

  @Override
  public String id() {
    return id;
  }

  @Override
  public String description() {
    return description;
  }

  public Alignment alignment() {
    return alignment;
  }

  public enum Alignment {
    DENSE, SPARSE, NULL
  }
}
