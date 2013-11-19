package com.spbsu.ml.loss;

import com.spbsu.commons.func.AdditiveGator;
import com.spbsu.commons.func.Factory;
import com.spbsu.ml.Oracle1;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 17:38
 */
public interface StatBasedOracle<T extends AdditiveGator> extends Oracle1 {
  Factory<T> statsFactory();
  double value(T comb);
  /** score MUST be additive to value :) */
  double score(T comb);
  double gradient(T comb);
}
