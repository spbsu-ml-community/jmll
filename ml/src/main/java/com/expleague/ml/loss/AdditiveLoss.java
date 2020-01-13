package com.expleague.ml.loss;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.func.Factory;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.TargetFunc;

import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 17:38
 */
public interface AdditiveLoss<T extends AdditiveStatistics> extends TargetFunc {
  IntFunction<T> statsFactory();

  int components();
  default IntStream nzComponents() {
    return IntStream.range(0, components());
  }
  double value(int component, double x);
  default double value(Vec x) {
    return nzComponents().parallel().mapToDouble(i -> value(i, x.get(i))).sum();
  }

  double value(T comb);
  /**
   * score MUST satisfy the value, at least correlate with it :)
   */
  double score(T comb);
  double bestIncrement(T comb);
}
