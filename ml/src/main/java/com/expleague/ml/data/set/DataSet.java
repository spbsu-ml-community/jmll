package com.expleague.ml.data.set;


import com.expleague.commons.func.CacheHolder;
import com.expleague.commons.func.ScopedCache;
import com.expleague.commons.seq.ArraySeq;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.SeqTools;
import com.expleague.ml.meta.DataSetMeta;

import java.util.stream.BaseStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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

    @Override
    public Seq<T> sub(int start, int end) {
      return new ArraySeq<>(this, start, end);
    }

    @Override
    public Seq<T> sub(int[] indices) {
      //noinspection unchecked
      return IntStream.of(indices).mapToObj(this::at).collect(SeqTools.collect((Class<T>)elementType())).build();
    }

    @Override
    public Stream<T> stream() {
      return IntStream.range(0, length()).mapToObj(this::at);
    }
  }
}
