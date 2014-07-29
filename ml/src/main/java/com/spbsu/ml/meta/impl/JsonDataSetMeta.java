package com.spbsu.ml.meta.impl;

import java.util.Date;


import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.meta.DataSetMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 13:26
 */
public class JsonDataSetMeta implements DataSetMeta {
  public Pool owner;
  public String source;
  public String author;
  public Date created;
  public ItemType type;
  public String id;

  public JsonDataSetMeta(final Pool owner, final String source, final String author, final Date created, final ItemType type,
                         final String id)
  {
    this.owner = owner;
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
  public Date created() {
    return created;
  }

  @Override
  public ItemType type() {
    return type;
  }
}
