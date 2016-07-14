package com.spbsu.crawl.bl;

import com.spbsu.crawl.bl.events.MapListener;
import com.spbsu.crawl.bl.events.HeroListener;
import com.spbsu.crawl.bl.events.TurnListener;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public interface GameSession {
  Hero.Race race();
  Hero.Spec spec();

  Mob.Action tryAction();

  Hero.Stat chooseStatForUpgrade();

}
