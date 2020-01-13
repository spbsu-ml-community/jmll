package com.expleague.ml.methods.greedyRegion;

import com.expleague.ml.loss.WeightedLoss;
import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.ml.loss.L2;

/**
 * Created by noxoomo on 10/11/14.
 */
public class AdditiveStatisticsExtractors {

  public static double sum(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat) {
      return ((L2.Stat) ((WeightedLoss.Stat) stats).inside).sum;
    }
    if (stats instanceof L2.Stat) {
      return ((L2.Stat) stats).sum;
    }
    return 0;
  }

  public static double sum2(final AdditiveStatistics stats) {
    if (stats instanceof WeightedLoss.Stat) {
      return ((L2.Stat) ((WeightedLoss.Stat) stats).inside).sum2;
    }
    if (stats instanceof L2.Stat) {
      return ((L2.Stat) stats).sum2;
    }
    return 0;
  }
}
