package com.spbsu.ml.data.softBorders.dataSet;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.VecDataSet;

/**
 * Created by noxoomo on 28/11/2016.
 */
public class SoftDataSet  {
  private final SoftGrid grid;
  private final VecDataSet base;
  private final int[][] points;
  private final byte[][] bins;

  public SoftDataSet(final VecDataSet base,
                     final SoftGrid grid) {
    this.base = base;
    this.grid = grid;
    points = new int[grid.rowsCount()][base.length()];
    bins = new byte[grid.rowsCount()][base.length()];
    final Mx data = base.data();
    for (int doc = 0; doc < data.rows(); ++doc) {
      for (int feature = 0; feature < data.columns(); ++feature) {
        points[feature][doc] = grid.row(feature).rank(data.get(doc, feature));
      }
    }
    for (int doc = 0; doc < data.rows(); ++doc) {
      for (int feature = 0; feature < data.columns(); ++feature) {
        bins[feature][doc] = grid.row(feature).bin(data.get(doc, feature));
      }
    }
  }

  public byte bin(final int feature,
                  final int docIdx) {
    return bins[feature][docIdx];
  }

  public Vec binDistribution(final int feature,
                             final int docIdx) {
    final int rk = points[feature][docIdx];
    return grid.row(feature).binsDistribution(rk);
  }

  public SoftGrid grid() {
    return grid;
  }

  VecDataSet base() {
    return base;
  }




}
