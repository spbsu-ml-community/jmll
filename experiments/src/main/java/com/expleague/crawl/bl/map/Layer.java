package com.expleague.crawl.bl.map;

import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.Optional;

/**
 * Created by noxoomo on 02/05/16.
 */

public class Layer<T> {
  private final TIntObjectHashMap<T> data = new TIntObjectHashMap<>();

  private int id(int x, int y) {
    return x + 100000 * y;
  }

  public void clear() {
    data.clear();
  }

  public Optional<T> item(final int x, final int y) {
    final int key = id(x, y);
    if (data.containsKey(key)) {
      return Optional.of(data.get(key));
    } else {
      return Optional.empty();
    }
  }

  public void setItem(final int x,
                      final int y,
                      final T type) {
    final int key = id(x, y);
    data.put(key, type);
  }

  public int itemCount() {
    return data.size();
  }
}
