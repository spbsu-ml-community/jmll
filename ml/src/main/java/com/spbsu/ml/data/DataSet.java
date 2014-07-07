package com.spbsu.ml.data;

import java.util.Date;


import com.spbsu.commons.func.CacheHolder;
import com.spbsu.commons.func.CacheHolderImpl;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.meta.PoolMeta;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:22
 */
public interface DataSet<T> extends Seq<T>, CacheHolder {
  PoolMeta meta();

  abstract class Stub<T> extends Seq.Stub<T> implements DataSet<T> {
    CacheHolder delegate = new CacheHolderImpl();
    Date creationDate = new Date();
    @Override
    public <CH extends CacheHolder, R> R cache(final Class<? extends Computable<CH, R>> type) {
      return delegate.cache(type);
    }
    @Override
    public PoolMeta meta() {
      return new PoolMeta() {
        @Override
        public String source() {
          return "/dev/random";
        }

        @Override
        public String author() {
          return "/dev/null";
        }

        @Override
        public Date created() {
          return creationDate;
        }
      };
    }

    @Override
    public boolean isImmutable() {
      return true;
    }
  }
}
