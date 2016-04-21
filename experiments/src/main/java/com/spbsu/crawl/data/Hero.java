package com.spbsu.crawl.data;

import com.spbsu.crawl.data.impl.KeyCode;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public class Hero {

  public enum Spec {
    Fighter(KeyCode.a);

    private final KeyCode select;
    Spec(KeyCode code) {
      select = code;
    }

    public KeyCode select() {
      return select;
    }
  }

  public enum Race {
    Minotaur(KeyCode.n);

    private final KeyCode select;
    Race(KeyCode code) {
      select = code;
    }

    public KeyCode select() {
      return select;
    }
  }
}
