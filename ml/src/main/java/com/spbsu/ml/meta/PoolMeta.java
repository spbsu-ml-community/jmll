package com.spbsu.ml.meta;

import java.util.Date;

/**
 * User: solar
 * Date: 04.07.14
 * Time: 15:11
 */
public interface PoolMeta {
  String source();
  String author();
  String file();
  Date created();

  class FakePoolMeta implements PoolMeta {
    private final Date creationDate = new Date();
    @Override
    public String source() {
      return "/dev/random";
    }

    @Override
    public String author() {
      return "/dev/null";
    }

    @Override
    public String file() {
      return "/dev/null";
    }

    @Override
    public Date created() {
      return creationDate;
    }

  };
}
