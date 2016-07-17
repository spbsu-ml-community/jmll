package com.spbsu.crawl.learning;

import com.spbsu.crawl.bl.events.StatusListener;
import com.spbsu.crawl.bl.helpers.CategoricalFeaturesMap;
import com.spbsu.crawl.learning.features.CategoricalFeature;
import com.spbsu.crawl.learning.features.Feature;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.stream.Stream;


/**
 * Created by noxoomo on 16/07/16.
 */
public class StatusFeaturesBuilder implements StatusListener, FeaturesBuilder {
  private final static CategoricalFeaturesMap statusIndex = new CategoricalFeaturesMap();
  private TIntSet currentStatus = new TIntHashSet();

  @Override
  public void addStatus(final String messages) {
    currentStatus.add(statusIndex.value(messages));
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
