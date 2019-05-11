package com.expleague.ml.meta.impl.fake;

import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.PoolTargetMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;

/**
 * User: qdeee
 * Date: 22.07.14
 */
public class FakeTargetMeta extends JsonTargetMeta implements PoolTargetMeta {
  public FakeTargetMeta(final ValueType valueType, int id) {
    super("Fake target #" + id, "Unknown source of target", valueType);
  }
}
