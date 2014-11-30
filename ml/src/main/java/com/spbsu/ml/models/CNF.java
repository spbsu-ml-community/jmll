package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.BitSet;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class CNF extends Func.Stub implements BinOptimizedModel {
  private final double inside;
  private final Clause[] clauses;
  private final BFGrid grid;

  public CNF(Clause[] conditions, double inside, BFGrid grid) {
    this.grid = grid;
    this.inside = inside;
    this.clauses = conditions;
  }

  @Override
  public double value(BinarizedDataSet bds, int pindex) {
    byte[] binarization = new byte[grid.rows()];
    for (int f = 0; f < grid.rows(); ++f) {
      binarization[f] = bds.bins(f)[pindex];
    }
    return value(binarization);
  }

  public double value(byte[] point) {
    return contains(point) ? inside : 0;
  }

  @Override
  public double value(Vec x) {
    byte[] binarizied = new byte[grid.rows()];
    grid.binarize(x, binarizied);
    return value(binarizied);
  }

  @Override
  public int dim() {
    return grid.rows();
  }

  public boolean contains(Vec x) {
    byte[] binarizied = new byte[grid.rows()];
    grid.binarize(x, binarizied);
    return contains(binarizied);
  }

  public boolean contains(byte[] point) {
    for (Clause clause : clauses) {
      if (!clause.contains(point)) {
        return false;
      }
    }
    return true;
  }


//
//  @Override
//  public boolean equals(Object o) {
//    if (this == o) return true;
//    if (!(o instanceof CherryRegion)) return false;
//    CherryRegion that = (CherryRegion) o;
//    if (!Arrays.equals(features, that.features)) return false;
//    if (!Arrays.equals(mask, that.mask)) return false;
//    if (this.inside != that.inside) return false;
//    if (this.outside != that.outside) return false;
//    if (this.maxFailed != that.maxFailed) return false;
//    if (this.score != that.score) return false;
//    if (this.basedOn != that.basedOn) return false;
//    return true;
//  }

//  public double score() {
//    return score;
//  }

  public static class Clause extends Func.Stub implements BinOptimizedModel {
    private final BFGrid grid;
    public Condition[] conditions;

    public Clause(BFGrid grid, Condition[] conditions) {
      this.grid = grid;
      this.conditions = conditions;
    }

    @Override
    public double value(BinarizedDataSet bds, int index) {
      byte[] binarization = new byte[grid.rows()];
      for (int f = 0; f < grid.rows(); ++f) {
        binarization[f] = bds.bins(f)[index];
      }
      return value(binarization);
    }

    public double value(byte[] point) {
      return contains(point) ? 1.0 : 0;
    }

    @Override
    public double value(Vec x) {
      byte[] binarizied = new byte[grid.rows()];
      grid.binarize(x, binarizied);
      return value(binarizied);
    }

    public boolean contains(byte[] point) {
      for (Condition condition : conditions) {
        if (condition.used.get(point[condition.feature])) {
          return true;
        }
      }
      return false;
    }

    @Override
    public int dim() {
      return grid.rows();
    }
  }

  public static class Condition implements Comparable<Condition> {
    public final int feature;
    public final BitSet used;

    public Condition(int feature, BitSet bins) {
      this.feature = feature;
      this.used = bins;
    }

    @Override
    public int compareTo(Condition o) {
      return Integer.compare(this.feature, o.feature);
    }
  }
}
