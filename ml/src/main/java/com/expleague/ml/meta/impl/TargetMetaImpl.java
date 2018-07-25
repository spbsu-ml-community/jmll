package com.expleague.ml.meta.impl;

import com.expleague.ml.meta.TargetMeta;

/**
 * User: solar
 * Date: 24.07.14
 * Time: 13:22
 */
public class TargetMetaImpl extends FeatureMetaImpl implements TargetMeta  {
  public TargetMetaImpl(final String id, final String description, final ValueType type) {
    super(id, description, type);
  }
}
