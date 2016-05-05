package com.spbsu.crawl.bl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.crawl.data.Action;

import java.util.stream.Stream;

/**
 * Experts League
 * Created by solar on 24/03/16.
 */
public interface Mob {
  Vec position();
  Stream<Action> actions();
}
