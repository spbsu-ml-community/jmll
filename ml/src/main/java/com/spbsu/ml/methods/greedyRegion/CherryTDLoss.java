package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.ml.data.CherryLoss;

public interface CherryTDLoss extends CherryLoss {
  double insideIncrement();

  double outsideIncrement();

  void nextIteration();

  double score();
}
