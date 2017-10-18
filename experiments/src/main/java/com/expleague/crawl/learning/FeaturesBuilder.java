package com.expleague.crawl.learning;

import com.expleague.crawl.bl.events.SystemViewListener;
import com.expleague.crawl.learning.features.Feature;

import java.util.stream.Stream;

/**
 * Created by noxoomo on 16/07/16.
 */
public interface FeaturesBuilder extends SystemViewListener {
  Stream<Feature> tickFeatures();
}
