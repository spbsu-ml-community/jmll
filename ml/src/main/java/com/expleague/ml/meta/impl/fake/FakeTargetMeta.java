package com.expleague.ml.meta.impl.fake;

import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.PoolTargetMeta;

/**
 * User: qdeee
 * Date: 22.07.14
 */
public class FakeTargetMeta extends FakeFeatureMeta implements PoolTargetMeta {
  public DataSet<?> ds;

  public FakeTargetMeta(final ValueType valueType, int id) {
    super(id, valueType);
  }

  @Override
  public String id() {
    return "Fake target #" + id;
  }
}
