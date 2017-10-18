package com.expleague.crawl.bl.crawlSystemView;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Created by noxoomo on 16/07/16.
 */

public interface Subscribable<T> {

  void subscribe(final T listener);

  abstract class Stub<T> implements Subscribable<T> {
    private List<T> listeners = new ArrayList<>();

    protected Stream<T> listeners() {
      return listeners.stream();
    }

    public void subscribe(final T listener) {
      listeners.add(listener);
    }

  }
}
