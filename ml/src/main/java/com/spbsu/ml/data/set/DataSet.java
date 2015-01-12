package com.spbsu.ml.data.set;


import com.spbsu.commons.func.CacheHolder;
import com.spbsu.commons.func.ScopedCache;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.meta.DataSetMeta;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:22
 */
public interface DataSet<Item> extends Seq<Item>, CacheHolder {
  DataSetMeta meta();
  int index(Item it);
  DataSet<?> parent();

  abstract class Stub<T> extends Seq.Stub<T> implements DataSet<T> {
    public ScopedCache cache = new ScopedCache(getClass(), this);
    private final DataSet<?> parent;

    protected Stub(final DataSet<?> parent) {
      this.parent = parent;
    }

    @Override
    public ScopedCache cache() {
      return cache;
    }

    @Override
    public DataSetMeta meta() {
      return parent.meta();
    }

    @Override
    public int index(final T obj) {
      for (int i = 0; i < length(); i++) {
        if (at(i).equals(obj))
          return i;
      }
      throw new RuntimeException("Object is not in dataset");
    }

    @Override
    public final boolean isImmutable() {
      return true;
    }

    @Override
    public final DataSet<?> parent() {
      return parent;
    }
  }
}
