package com.spbsu.ml.data.cherry;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.BFGrid;

/**
 * User: noxoomo
 * Date: 25.03.15
 */

public interface CherryLoss  {
  double score(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out);
  double score();
  double insideIncrement();
  default double outsideIncrement() {
    return 0;
  }

  default void addCondition(BFGrid.BFRow feature, int start, int end) {
    subset().addCondition(feature,start,end);
  }

  default void startClause() {
    subset().startClause();
  }

  default void endClause() {
    subset().endClause();
  }

  CherryPointsHolder subset();
}
