package com.spbsu.ml.models.pgm;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;

/**
 * User: solar
 * Date: 07.04.14
 * Time: 21:36
 */
public interface Route {
  double p();
  int last();
  int length();

  ProbabilisticGraphicalModel dstOwner(int stepNo);
  int dst(int stepNo);
}
