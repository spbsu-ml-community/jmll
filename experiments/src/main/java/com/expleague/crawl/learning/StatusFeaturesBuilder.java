package com.expleague.crawl.learning;

import com.expleague.crawl.bl.events.StatusListener;
import com.expleague.crawl.bl.helpers.CategoricalFeaturesMap;
import com.expleague.crawl.learning.features.CategoricalFeature;
import com.expleague.crawl.learning.features.Feature;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.stream.Stream;


/**
 * Created by noxoomo on 16/07/16.
 */
public class StatusFeaturesBuilder implements StatusListener, FeaturesBuilder {
  private final static CategoricalFeaturesMap statusIndex = new CategoricalFeaturesMap(
//          "Water",
//          "Hungry",
//          "Very Hungry",
//          "Near Starving",
//          "Starving",
//          "Fainting",
//          "Para",
//          "Constr",
//          "Drain",
//          "Held",
//          "Pois",
//          "Slow"
  );
  private TIntSet currentStatus = new TIntHashSet();

  public StatusFeaturesBuilder(final CategoricalFeaturesMap statusIndex) {
//    this.statusIndex = statusIndex;
  }

  @Override
  public void addStatus(final String status) {
    final int value = statusIndex.value(status);
    if (value < 0)
      System.out.println("New status: " + status);
    else
      currentStatus.add(value);
  }

  @Override
  public void removeStatus(final String messages) {
    currentStatus.remove(statusIndex.value(messages));
  }


  @Override
  public Stream<Feature> tickFeatures() {
    return Stream.of(new CategoricalFeature(statusIndex, new TIntHashSet(currentStatus), "status"));
  }
}
