package com.expleague.ml.meta.impl.fake;

import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 16:56
 */
public class FakeFeatureMeta extends JsonFeatureMeta implements FeatureMeta {
  public FakeFeatureMeta(int id, ValueType type) {
    super("Fake feature #" + id, "Unknown source of feature", type);
  }
}
