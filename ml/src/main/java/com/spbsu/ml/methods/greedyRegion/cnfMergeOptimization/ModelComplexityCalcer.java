package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.models.CNF;

import java.util.BitSet;

/**
 * Created by noxoomo on 04/12/14.
 */
class ModelComplexityCalcer {
  private final BFGrid grid;
  private final int[][] base;
  BitSet[] used;
  public double total;

  public ModelComplexityCalcer(BinarizedDataSet bds, int[] points, BitSet[] used) {
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


  public double calculate(CNF.Clause clause) {
    double reg = 1;//Double.POSITIVE_INFINITY;
    for (CNF.Condition condition : clause.conditions) {
      double information = 0;
      double count = 0;
      int f = condition.feature;
      boolean current = false;
      int cardinality = 0;
      if (grid.row(f).size() == 1)
        return 0;
      for (int bin = 0; bin <= grid.row(f).size(); ++bin) {
        if (condition.used.get(bin) && used[f].get(bin)) { //don't use features, which was used on previous levels
//            return Double.POSITIVE_INFINITY;
          return 0;
        }
        if (condition.used.get(bin) == current) {
          count += base[f][bin];
        } else {
          information += count > 0 ? count * Math.log(count) : 0;
          ++cardinality;
//            information += Math.log(count + 1);
          current = condition.used.get(bin);
          count = base[f][bin];
        }
        if (current && base[f][bin] == 0)
          return 0;
      }
      ++cardinality;
      information += count > 0 ? count * Math.log(count) : 0;
//        information += Math.log(count + 1);
      information /= total;
      double entropy = Math.log(total) - information;
      double normalized = -entropy / Math.log(1.0 / cardinality);
//        reg += information;
      reg = Math.min(normalized, reg);//Math.min(information, reg);
    }
    return -reg;//reg / layer.conditions.length;
  }


  public static double cardinality(BFGrid grid, CNF.Clause clause) {
    double cardinality = 0;
    for (CNF.Condition condition : clause.conditions) {
      double count = 0;
      int f = condition.feature;
      for (int bin = 0; bin <= grid.row(f).size(); ++bin) {
        if (condition.used.get(bin)) {
          count = 1;
        } else {
          cardinality += count;
          count = 0;
        }
      }
      cardinality += count;
    }
    return cardinality + 2 * clause.conditions.length;
  }
}
