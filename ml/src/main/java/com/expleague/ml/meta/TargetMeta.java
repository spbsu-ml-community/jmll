package com.expleague.ml.meta;

import com.expleague.ml.meta.impl.TargetMetaImpl;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 15:53
 */
public interface TargetMeta extends FeatureMeta {
  static TargetMeta create(final String id, final String description, final ValueType type) {
    return new TargetMetaImpl(id, description, type);
  }
}
