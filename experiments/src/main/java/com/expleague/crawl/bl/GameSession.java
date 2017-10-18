package com.expleague.crawl.bl;

import com.expleague.crawl.bl.events.SystemViewListener;
import com.expleague.crawl.learning.features.Feature;

import java.util.List;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public interface GameSession extends SystemViewListener {
  Hero.Race race();
  Hero.Spec spec();

  Mob.Action action();

  Hero.Stat chooseStatForUpgrade();

  void features(List<Feature> features);
  void finish();
}
