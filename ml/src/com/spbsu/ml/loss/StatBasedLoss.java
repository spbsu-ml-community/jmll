package com.spbsu.ml.loss;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.ml.Func;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 17:38
 */
public interface StatBasedLoss<T extends AdditiveStatistics> extends Func {
  Factory<T> statsFactory();
  double value(T comb);
  /** score MUST be additive to value :) */
  double score(T comb);
  double bestIncrement(T comb);
}
