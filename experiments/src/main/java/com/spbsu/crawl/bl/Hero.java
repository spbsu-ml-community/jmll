package com.spbsu.crawl.bl;

/**
 * Experts League
 * Created by solar on 24/03/16.
 */
public interface Hero extends Mob {
  Body body();

  Action act(Situation situation);

  interface Body {
  }
}
