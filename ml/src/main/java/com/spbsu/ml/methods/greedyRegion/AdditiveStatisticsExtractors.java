package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;

/**
 * Created by noxoomo on 10/11/14.
 */
public class AdditiveStatisticsExtractors {
  public static double weight(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat) {
      return ((L2.MSEStats) ((WeightedLoss.Stat) stats).inside).weight;
    }
    if (stats instanceof L2.MSEStats) {
      return ((L2.MSEStats) stats).weight;
    }
    return 0;
  }

  public static double sum(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat) {
      return ((L2.MSEStats) ((WeightedLoss.Stat) stats).inside).sum;
    }
    if (stats instanceof L2.MSEStats) {
      return ((L2.MSEStats) stats).sum;
    }
    return 0;
  }

  public static double sum2(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat) {
      return ((L2.MSEStats) ((WeightedLoss.Stat) stats).inside).sum2;
    }
    if (stats instanceof L2.MSEStats) {
      return ((L2.MSEStats) stats).sum2;
    }
    return 0;
  }
}
