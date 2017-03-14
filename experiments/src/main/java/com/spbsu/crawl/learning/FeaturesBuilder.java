package com.spbsu.crawl.learning;

import com.spbsu.crawl.bl.events.SystemViewListener;
import com.spbsu.crawl.learning.features.Feature;

import java.util.stream.Stream;

/**
 * Created by noxoomo on 16/07/16.
 */
public interface FeaturesBuilder extends SystemViewListener {
  Stream<Feature> tickFeatures();
}
