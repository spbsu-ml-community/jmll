package com.expleague.crawl.learning;

import com.expleague.crawl.bl.events.HeroListener;
import com.expleague.crawl.learning.features.Feature;
import com.expleague.crawl.learning.features.NumericalFeature;

import java.util.stream.Stream;

/**
 * Created by noxoomo on 17/07/16.
 */
public class HeroFeaturesBuilder implements FeaturesBuilder, HeroListener {
  int hp = 0;
  int x = 0;
  int y = 0;

  @Override
  public void heroPosition(final int x, final int y) {
    this.x = x;
    this.y = y;
  }

  @Override
  public void hp(final int hp) {
    this.hp = hp;
  }

  @Override
  public Stream<Feature> tickFeatures() {
    return Stream.of(
            new NumericalFeature(hp, "Health"),
            new NumericalFeature(x, "x"),
            new NumericalFeature(y, "y"));
  }
}
