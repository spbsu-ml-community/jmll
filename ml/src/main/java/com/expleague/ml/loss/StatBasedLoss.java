package com.expleague.ml.loss;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.func.Factory;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.TargetFunc;

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
