package com.spbsu.crawl.bl;

import com.spbsu.commons.math.vectors.Vec;

import java.util.stream.Stream;

/**
 * Experts League
 * Created by solar on 24/03/16.
 */
public interface Situation {
  Map map();
  Stream<Event> history();
  Mob[] mobs();
}
