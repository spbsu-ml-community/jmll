package com.expleague.ml.meta.impl.fake;

import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.PoolFeatureMeta;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 16:56
 */
public class FakeFeatureMeta implements PoolFeatureMeta {
  protected final int id;
  private final ValueType type;
  private Pool owner;

  public FakeFeatureMeta(int id, ValueType type) {
    this.id = id;
    this.type = type;
  }

  @Override
  public String id() {
    return "Fake feature #" + id;
  }

  @Override
  public String description() {
    return "Unknown source of feature";
  }

  @Override
  public ValueType type() {
    return type;
  }

  @Override
  public <T extends DSItem> Pool<T> owner() {
    //noinspection unchecked
    return this.owner;
  }

  @Override
  public void setOwner(Pool<? extends DSItem> owner) {
    this.owner = owner;
  }
}
