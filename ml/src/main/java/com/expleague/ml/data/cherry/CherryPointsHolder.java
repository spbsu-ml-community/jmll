package com.expleague.ml.data.cherry;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.ml.data.Aggregate;
import com.expleague.ml.BFGrid;
import com.expleague.ml.impl.BFRowImpl;

/**
 * Created by noxoomo on 06/04/15.
 */
public interface CherryPointsHolder  {

  void visitAll(final Aggregate.IntervalVisitor<? extends AdditiveStatistics> visitor);

  BFGrid grid();

  void endClause();
  void startClause();

  AdditiveStatistics addCondition(BFGrid.Row feature, int start, int end);

  AdditiveStatistics inside();
  AdditiveStatistics outside();

}
