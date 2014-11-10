package com.spbsu.ml.loss;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.TargetFunc;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 17:38
 */
public interface StatBasedLoss<T extends AdditiveStatistics> extends TargetFunc {
  Factory<T> statsFactory();

  Vec target();

  double value(T comb);

  /**
   * score MUST be additive to value :)
   */
  double score(T comb);

  double bestIncrement(T comb);
}
