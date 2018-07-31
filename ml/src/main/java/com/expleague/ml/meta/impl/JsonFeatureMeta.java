package com.expleague.ml.meta.impl;

import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.PoolFeatureMeta;
import com.expleague.ml.data.set.DataSet;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 13:44
 */
public class JsonFeatureMeta extends FeatureMeta.Stub implements PoolFeatureMeta {
  @JsonProperty
  private String id;
  @JsonProperty
  private String description;
  @JsonProperty
  private ValueType type;
  @JsonProperty
  private String associated;
  private transient Pool owner;

  public JsonFeatureMeta(FeatureMeta meta, String associated) {
    this.id = meta.id();
    this.description = meta.description();
    this.type = meta.type();
    this.associated = associated;
  }

  public JsonFeatureMeta() {
  }

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

  @Override
  public <T extends DSItem> Pool<T> owner() {
    //noinspection unchecked
    return (Pool<T>) owner;
  }

  @Override
  public void setOwner(Pool<? extends DSItem> result) {
    this.owner = result;
  }

  @Override
  public DataSet<?> associated() {
    return owner.data(associated);
  }
}
