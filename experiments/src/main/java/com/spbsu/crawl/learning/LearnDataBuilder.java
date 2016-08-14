package com.spbsu.crawl.learning;

import com.spbsu.commons.util.Pair;
import com.spbsu.crawl.bl.Mob;
import com.spbsu.crawl.bl.crawlSystemView.SystemView;
import com.spbsu.crawl.bl.events.PlayerActionListener;
import com.spbsu.crawl.bl.helpers.CategoricalFeaturesMap;
import com.spbsu.crawl.learning.features.Feature;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class LearnDataBuilder implements PlayerActionListener {
  private static final String STATUS_DICT_FILE = "status_keys.json";
  private final CategoricalFeaturesMap catFeaturesMap;
  private final List<Pair<State, Mob.Action>> session = new ArrayList<>();
  private final List<FeaturesBuilder> builders = new ArrayList<>();

  public LearnDataBuilder() throws IOException {
    builders.add(new InventoryFeaturesBuilder());
    File statusFeaturesJsonFile = new File(STATUS_DICT_FILE);
    if (statusFeaturesJsonFile.isFile()) {
      catFeaturesMap = CategoricalFeaturesMap.load(statusFeaturesJsonFile);
    } else {
      catFeaturesMap = new CategoricalFeaturesMap();
    }
    builders.add(new StatusFeaturesBuilder(catFeaturesMap));
    builders.add(new HeroFeaturesBuilder());
  }

  @Override
  public void action(final Mob.Action action) {
    final List<Feature> features = builders.stream().flatMap(FeaturesBuilder::tickFeatures).collect(Collectors.toList());
    session.add(Pair.create(new State(features), action));
  }

  public void attach(final SystemView view) {
    builders.forEach(view::subscribe);
    view.subscribe(this);
  }

  public List<Pair<State, Mob.Action>> session() {
    return session;
  }

  public void endGame() {
    catFeaturesMap.save(new File(STATUS_DICT_FILE));
  }
}
