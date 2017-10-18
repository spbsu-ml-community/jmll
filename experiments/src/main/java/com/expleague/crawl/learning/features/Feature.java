package com.expleague.crawl.learning.features;

/**
 * Created by noxoomo on 17/07/16.
 */
public interface Feature {
  int dim();
  int at(int i);
  String name();
}
