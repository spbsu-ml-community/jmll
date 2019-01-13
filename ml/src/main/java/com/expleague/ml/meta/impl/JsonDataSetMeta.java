package com.expleague.ml.meta.impl;

import java.util.Date;
import java.util.Objects;


import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.DataSetMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 13:26
 */
public class JsonDataSetMeta implements DataSetMeta {
  public String id;
  public Pool owner;
  public String source;
  public String author;
  public Date created;
  public Class<?> type;

  public JsonDataSetMeta(final String source, final String author, final Date created, final Class<?> type, final String id) {
    this.source = source;
    this.author = author;
    this.created = created;
    this.type = type;
    this.id = id;
  }

  public JsonDataSetMeta() {
  }

  @Override
  public String id() {
    return id;
  }

  @Override
  public String source() {
    return source;
  }

  @Override
  public String author() {
    return author;
  }

  @Override
  public Pool owner() {
    return owner;
  }

  @Override
  public void setOwner(Pool pool) {
    owner = pool;
  }

  @Override
  public Date created() {
    return created;
  }

  @Override
  public Class<?> type() {
    return type;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o)
      return true;
    if (!(o instanceof JsonDataSetMeta))
      return false;
    JsonDataSetMeta that = (JsonDataSetMeta) o;
    return id.equals(that.id) && owner.equals(that.owner) && source.equals(that.source) && author.equals(that.author) && created.equals(that.created) && type.equals(that.type);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, owner, source, author, created, type);
  }
}
