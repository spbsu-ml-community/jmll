package com.spbsu.crawl.learning.features;

import com.spbsu.commons.math.vectors.Vec;

/**
 * Created by noxoomo on 17/07/16.
 */
public interface Feature {
  int dim();
  int at(int i);
  String name();
}
