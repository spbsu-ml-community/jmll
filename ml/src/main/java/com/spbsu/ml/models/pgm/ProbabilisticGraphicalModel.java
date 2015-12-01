package com.spbsu.ml.models.pgm;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;

/**
 * User: solar
 * Date: 07.04.14
 * Time: 21:34
 */
public interface ProbabilisticGraphicalModel extends Trans, Func {
  void visit(Filter<Route> act);

  double p(int... controlPoints);

  @Override
  double value(Vec x);

  @Override
  int dim();

  int knownRoutesCount();
  Route knownRoute(int index);
  double knownRoutesWeight();

  Route next(FastRandom rng);

  boolean isFinal(int node);
}
