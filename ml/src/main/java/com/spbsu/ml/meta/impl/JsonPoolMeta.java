package com.spbsu.ml.meta.impl;

import java.util.Date;


import com.fasterxml.jackson.core.JsonParser;
import com.spbsu.ml.meta.PoolMeta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 13:26
 */
public class JsonPoolMeta implements PoolMeta {
  private final String file;
  public String source;
  public String author;
  public Date created;
  public boolean duplicates;

  public JsonPoolMeta(String file) {
    this.file = file;
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
  public String file() {
    return file;
  }

  @Override
  public Date created() {
    return created;
  }

  @Override
  public boolean duplicatesAllowed() {
    return duplicates;
  }
}
