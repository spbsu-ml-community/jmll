package com.spbsu.crawl.bl;

import com.spbsu.crawl.bl.events.SystemViewListener;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public interface GameSession extends SystemViewListener {
  Hero.Race race();
  Hero.Spec spec();

  Mob.Action action();

  Hero.Stat chooseStatForUpgrade();

}
