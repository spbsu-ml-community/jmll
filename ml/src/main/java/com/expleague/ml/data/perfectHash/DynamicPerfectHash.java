package com.expleague.ml.data.perfectHash;

import com.expleague.commons.func.WeakListenerHolder;
import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import gnu.trove.map.hash.TDoubleIntHashMap;

public interface DynamicPerfectHash<U> extends PerfectHash<U> {
  int add(U value);

  public abstract class Stub<U> extends WeakListenerHolderImpl<Integer> implements DynamicPerfectHash<U>  {
    final private TDoubleIntHashMap hash;

    protected Stub() {
      this.hash = new TDoubleIntHashMap();
    }

    protected abstract double key(final U entry);

    @Override
    public int id(final U value) {
      final double key = key(value);
      return hash.containsKey(key) ? hash.get(key) : -1;
    }

    @Override
    public int size() {
      return hash.size();
    }

    @Override
    public int add(final U value) {
      final double key = key(value);
      if (!hash.containsKey(key)) {
        final int newHash = hash.size();
        hash.put(key, hash.size());
        invoke(newHash);
        return newHash;
      }
      return hash.get(key);
    }
  }

}
