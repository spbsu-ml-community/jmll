package com.spbsu.ml.data.cherry;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregate;

/**
 * Created by noxoomo on 06/04/15.
 */
public interface CherryPointsHolder  {

  void visitAll(final Aggregate.IntervalVisitor<? extends AdditiveStatistics> visitor);

  BFGrid grid();

  void endClause();
  void startClause();

  AdditiveStatistics addCondition(BFGrid.BFRow feature, int start, int end);

  AdditiveStatistics inside();
  AdditiveStatistics outside();

}
