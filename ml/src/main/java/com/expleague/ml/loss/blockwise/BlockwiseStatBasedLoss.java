package com.expleague.ml.loss.blockwise;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.func.Factory;
import com.expleague.ml.BlockwiseFuncC1;
import com.expleague.ml.TargetFunc;

/**
 * User: solar
 * Date: 12.11.13
 * Time: 17:38
 */
public interface BlockwiseStatBasedLoss<T extends AdditiveStatistics> extends BlockwiseFuncC1, TargetFunc {
  Factory<T> statsFactory();
  double value(T comb);
  /** score MUST be additive to value :) */
  double score(T comb);
  double bestIncrement(T comb);
}
