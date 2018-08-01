package com.expleague.ml.data.cherry;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.ml.BFGrid;
import com.expleague.ml.impl.BFRowImpl;

/**
 * User: noxoomo
 * Date: 25.03.15
 */

public abstract class CherryLoss {
  public abstract double score(BFGrid.Row feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out);

  public abstract double score();

  public abstract double insideIncrement();

   public double outsideIncrement() {
    return 0;
  }

  public void addCondition(BFGrid.Row feature, int start, int end) {
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
