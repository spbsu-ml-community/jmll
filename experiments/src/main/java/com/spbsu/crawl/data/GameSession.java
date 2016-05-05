package com.spbsu.crawl.data;

import com.spbsu.crawl.bl.Mob;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public interface GameSession extends MapListener {
  Hero.Race race();
  Hero.Spec spec();

  void heroPosition(int x, int y);

  Mob.Action tick();

  Hero.Stat chooseStatForUpgrade();

}
