package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.BFGrid;

/**
 * User: noxoomo
 * Date: 25.03.15
 */

public interface CherryLoss  {
  double score(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out);
  void added(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added);
}
