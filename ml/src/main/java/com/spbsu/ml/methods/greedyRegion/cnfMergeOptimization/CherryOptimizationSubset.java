package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.models.CNF;
import gnu.trove.list.array.TIntArrayList;

/**
 * Created by noxoomo on 30/11/14.
 */
public class CherryOptimizationSubset {
  CNF.Clause clause;
  BinarizedDataSet bds;
  public int[] outside;
  public int[] inside;
  public AdditiveStatistics stat;
  public double regularization = Double.POSITIVE_INFINITY;
  public boolean isRegularizationKnown = false;

  CherryOptimizationSubset(BinarizedDataSet bds, CNF.Clause clause, int[] inside, int[] outside, AdditiveStatistics stat) {
    this.bds = bds;
    this.clause = clause;
    this.inside = inside;
    this.outside = outside;
    this.stat = stat;
  }

  public CherryOptimizationSubset(BinarizedDataSet bds, AdditiveStatistics stat, CNF.Clause clause, int[] points) {
    TIntArrayList insidePoints = new TIntArrayList();
    TIntArrayList outsidePoints = new TIntArrayList();
    for (int i = 0; i < points.length; ++i) {
      if (clause.value(bds, points[i]) == 1.0) {
        stat.append(points[i], 1);
        insidePoints.add(points[i]);
      } else {
        outsidePoints.add(points[i]);
      }
    }
    this.outside = outsidePoints.toArray();
    this.inside = insidePoints.toArray();
    this.stat = stat;
    this.bds = bds;
    this.clause = clause;
  }
}
