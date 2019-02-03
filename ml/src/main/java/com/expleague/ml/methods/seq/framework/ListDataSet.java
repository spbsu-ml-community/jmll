package com.expleague.ml.methods.seq.framework;

import com.expleague.ml.data.set.DataSet;

import java.util.List;

/**
 * Created by hrundelb on 02.02.19.
 */
public class ListDataSet<T> extends DataSet.Stub<T> {

  private final List<T> list;

  public ListDataSet(List<T> list) {
    super(null);
    this.list = list;
  }

  @Override
  public T at(int i) {
    return list.get(i);
  }

  @Override
  public int length() {
    return list.size();
  }

  @SuppressWarnings("unchecked")
  @Override
  public Class<? extends T> elementType() {
    return (Class<T>) list.get(0).getClass();
  }
}
