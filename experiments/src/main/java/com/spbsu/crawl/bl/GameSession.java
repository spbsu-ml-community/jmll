package com.spbsu.crawl.bl;

import com.spbsu.crawl.bl.events.SystemViewListener;
import com.spbsu.crawl.learning.features.Feature;

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
