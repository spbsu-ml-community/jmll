package com.spbsu.ml.models.pgm;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;

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
  double knownRouteWeight();

  boolean isFinal(int node);
}
