package com.expleague.ml.meta.impl;

import com.expleague.ml.meta.PoolTargetMeta;
import com.expleague.ml.meta.TargetMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 13:44
 */
public class JsonTargetMeta extends JsonFeatureMeta implements PoolTargetMeta {
  public JsonTargetMeta(TargetMeta meta, String associated) {
    super(meta, associated);
  }

  public JsonTargetMeta() {
  }
}
