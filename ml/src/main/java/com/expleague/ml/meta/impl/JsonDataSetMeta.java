package com.expleague.ml.meta.impl;

import java.util.Date;


import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.DataSetMeta;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

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
  public boolean equals(final Object obj) {
    if (obj == this)
      return true;
    if (obj == null || obj.getClass() != getClass())
      return false;

    final JsonDataSetMeta other = (JsonDataSetMeta)obj;
    return new EqualsBuilder().
        append(id, other.id).
        append(source, other.source).
        append(author, other.author).
        append(created, other.created).
        append(type, other.type).
        isEquals();
  }

  @Override
  public int hashCode() {
    return new HashCodeBuilder().
        append(id).
        append(source).
        append(author).
        append(created).
        append(type).
        toHashCode();
  }
}
