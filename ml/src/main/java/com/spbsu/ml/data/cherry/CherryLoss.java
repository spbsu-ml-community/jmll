package com.spbsu.ml.data.cherry;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.BFGrid;

/**
 * User: noxoomo
 * Date: 25.03.15
 */

public abstract class CherryLoss {
  public abstract double score(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out);

  public abstract double score();

  public abstract double insideIncrement();

   public double outsideIncrement() {
    return 0;
  }

  public void addCondition(BFGrid.BFRow feature, int start, int end) {
    subset().addCondition(feature,start,end);
  }

  public void startClause() {
    subset().startClause();
  }

  public void endClause() {
    subset().endClause();
  }

  public abstract CherryPointsHolder subset();
}
