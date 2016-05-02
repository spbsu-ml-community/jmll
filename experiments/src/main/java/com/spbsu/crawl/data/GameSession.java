package com.spbsu.crawl.data;

import com.spbsu.crawl.bl.Mob;
import com.spbsu.crawl.bl.map.mapEvents.MapEvent;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public interface GameSession {
  Hero.Race race();
  Hero.Spec spec();

  void heroPosition(int x, int y);

  Mob.Action tick();

  void updateMap(MapEvent event);
}
