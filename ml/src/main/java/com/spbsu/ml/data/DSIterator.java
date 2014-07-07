package com.spbsu.ml.data;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.meta.DSItem;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:26
 */
public interface DSIterator {
  boolean advance();

  double y();
  double x(int i);
  Vec x();

  DSItem item();

  int index();
}
