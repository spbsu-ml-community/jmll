package com.expleague.ml.models.pgm;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;

import java.util.function.Predicate;

/**
 * User: solar
 * Date: 07.04.14
 * Time: 21:34
 */
public interface ProbabilisticGraphicalModel extends Trans, Func {
  void visit(Predicate<com.expleague.ml.models.pgm.Route> act);

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
