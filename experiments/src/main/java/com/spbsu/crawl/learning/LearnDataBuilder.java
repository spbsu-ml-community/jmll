package com.spbsu.crawl.learning;

import com.spbsu.commons.util.Pair;
import com.spbsu.crawl.bl.Mob;
import com.spbsu.crawl.bl.crawlSystemView.SystemView;
import com.spbsu.crawl.bl.events.PlayerActionListener;
import com.spbsu.crawl.learning.features.Feature;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class LearnDataBuilder implements PlayerActionListener {
  List<Pair<State, Mob.Action>> session = new ArrayList<>();
  List<FeaturesBuilder> builders = new ArrayList<>();

  public LearnDataBuilder() {
    builders.add(new InventoryFeaturesBuilder());
    builders.add(new StatusFeaturesBuilder());
    builders.add(new HeroFeaturesBuilder());
  }

  @Override
  public void action(final Mob.Action action) {
    List<Feature> features = builders.stream().flatMap(FeaturesBuilder::tickFeatures).collect(Collectors.toList());
    session.add(Pair.create(new State(features), action));
  }


  public void attach(final SystemView view) {
    builders.forEach(view::subscribe);
    view.subscribe(this);
  }
}
