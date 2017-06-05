package com.spbsu.crawl.learning;

import com.spbsu.crawl.bl.Mob;
import com.spbsu.crawl.bl.events.PlayerActionListener;
import com.spbsu.crawl.learning.features.Feature;
import com.spbsu.crawl.learning.features.NumericalFeature;

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
