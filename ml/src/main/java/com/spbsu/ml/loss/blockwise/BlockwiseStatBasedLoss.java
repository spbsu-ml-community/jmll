package com.spbsu.ml.loss.blockwise;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.ml.BlockwiseFuncC1;
import com.spbsu.ml.TargetFunc;

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
