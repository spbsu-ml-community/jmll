package com.expleague.crawl.learning;

import com.expleague.crawl.bl.Mob;
import com.expleague.crawl.bl.events.PlayerActionListener;
import com.expleague.crawl.learning.features.Feature;
import com.expleague.crawl.learning.features.NumericalFeature;

import java.util.stream.Stream;

/**
 * Experts League
 * Created by solar on 28.07.16.
 */
public class TickFeatureBuilder implements FeaturesBuilder, PlayerActionListener {
  int counter = 0;
  @Override
  public Stream<Feature> tickFeatures() {
    return Stream.of(new NumericalFeature(counter, "Time"));
  }

  @Override
  public void action(Mob.Action action) {
    counter++;
  }
}
