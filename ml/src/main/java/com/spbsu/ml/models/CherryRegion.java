package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinModelWithGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.BitSet;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class CherryRegion extends BinOptimizedModel.Stub implements BinModelWithGrid {
  public final double inside;
  private final double outside = 0;
  private final BitSet[] conditions;
  private final BFGrid grid;
  private final int[] starts;

  public CherryRegion(final BitSet[] conditions, final double inside, final BFGrid grid) {
    this.conditions = conditions;
    this.grid = grid;
    this.inside = inside;

    this.starts = new int[grid.rows()];
    int tmp = 0;
    for (int i = 0; i < grid.rows(); i++) {
      starts[i] = tmp;
      tmp += grid.row(i).size() + 1;
    }
  }

  @Override
  public BFGrid grid() {
    return grid;
  }

  @Override
  public double value(final BinarizedDataSet bds, final int pindex) {
    final byte[] binarization = new byte[grid.rows()];
    for (int f = 0; f < grid.rows(); ++f) {
      binarization[f] = bds.bins(f)[pindex];
    }
    return value(binarization);
  }

  public double value(final byte[] point) {

    return contains(point) ? inside : outside;

  }

  @Override
  public double value(final Vec x) {
    final byte[] binarizied = new byte[grid.rows()];
    grid.binarize(x, binarizied);
    return value(binarizied);
  }

  @Override
  public int dim() {
    return grid.rows();
  }

  public boolean contains(final Vec x) {
    final byte[] binarizied = new byte[grid.rows()];
    grid.binarize(x, binarizied);
    return contains(binarizied);
  }

  public boolean contains(final byte[] point) {
    boolean contains = true;
    for (final BitSet condition : conditions) {
      boolean good = false;
      for (int i = 0; i < grid.rows(); ++i) {
        good = good | condition.get(starts[i] + point[i]);
      }
      contains &= good;
      if (!contains) {
        break;
      }
    }
    return contains;
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
}
