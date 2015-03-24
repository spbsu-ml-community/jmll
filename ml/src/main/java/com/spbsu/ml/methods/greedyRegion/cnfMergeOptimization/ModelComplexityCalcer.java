package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;


import java.util.BitSet;

/**
 * Created by noxoomo on 04/12/14.
 */
class ModelComplexityCalcer {
  private final BFGrid grid;
  private final int[][] base;
  BitSet[] used;
  public double total;

  public ModelComplexityCalcer(final BinarizedDataSet bds, final int[] points, final BitSet[] used) {
    this.grid = bds.grid();
    this.used = used;
    base = new int[grid.rows()][];
    {
      for (int feature = 0; feature < grid.rows(); feature++) {
        base[feature] = new int[grid.row(feature).size() + 1];
        final byte[] bin = bds.bins(feature);
        for (int j = 0; j < points.length; j++) {
          base[feature][bin[points[j]]]++;
        }
      }
    }
    total = 0;
    for (int bin = 0; bin <= grid.row(0).size(); ++bin) {
      total += base[0][bin];
    }
  }
}
