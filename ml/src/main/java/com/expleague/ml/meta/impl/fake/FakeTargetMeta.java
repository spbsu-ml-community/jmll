package com.expleague.ml.meta.impl.fake;

import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 22.07.14
 */
public class FakeTargetMeta extends FakeFeatureMeta implements TargetMeta {
  public DataSet<?> ds;

  public FakeTargetMeta(final ValueType valueType, int id) {
    super(id, valueType);
  }

  @Override
  public String id() {
    return "Fake target #" + id;
  }
}
