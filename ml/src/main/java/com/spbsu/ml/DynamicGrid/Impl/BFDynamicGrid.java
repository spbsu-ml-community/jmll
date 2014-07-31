package com.spbsu.ml.DynamicGrid.Impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
import com.spbsu.ml.DynamicGrid.Interface.DynamicRow;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.stats.OrderByFeature;
import gnu.trove.set.hash.TIntHashSet;

public class BFDynamicGrid implements DynamicGrid {
  private DynamicRow[] rows;
  private TIntHashSet known = new TIntHashSet();
  private final DynamicRow leastNonEmptyRow;

  public BFDynamicGrid(VecDataSet ds, int minSplits) {
    final OrderByFeature byFeature = ds.cache().cache(OrderByFeature.class, DataSet.class);

    MedianRow[] rows = new MedianRow[ds.data().columns()];
    for (int f = 0; f < ds.data().columns(); ++f) {
      final ArrayPermutation permutation = byFeature.orderBy(f);
      int[] order = permutation.direct();
      double[] feature = new double[order.length];
      for (int i = 0; i < feature.length; i++)
        feature[i] = ds.at(order[i]).get(f);
      rows[f] = new MedianRow(this, feature, permutation.reverse(), f, minSplits);
    }
    DynamicRow least = null;
    for (int f = 0; f < rows.length; ++f)
      if (!rows[f].empty()) {
        least = rows[f];
        break;
      }
    this.rows = rows;
    leastNonEmptyRow = least;

  }


  public DynamicRow row(int feature) {
    return feature < rows.length ? rows[feature] : null;
  }

  @Override
  public void binarize(Vec x, int[] folds) {
    for (int i = 0; i < x.dim(); i++) {
      folds[i] = rows[i].bin(x.get(i));
    }

  }

  @Override
  public BinaryFeature bf(int fIndex, int binNo) {
    return rows[fIndex].bf(binNo);
  }

  @Override
  public DynamicRow nonEmptyRow() {
    return leastNonEmptyRow;
  }

  @Override
  public boolean addSplit(int feature) {
    return rows[feature].addSplit();
  }


  @Override
  public int[] hist() {
    int[] counts = new int[rows.length];
    for (int f = 0; f < rows.length; ++f) {
      counts[f] = rows[f].size();
    }
    return counts;
  }

  public int rows() {
    return rows.length;
  }

  @Override
  public void setKnown(int hash) {
    known.add(hash);
  }

  @Override
  public boolean isKnown(int hash) {
    return known.contains(hash);
  }


  @Override
  public boolean isActive(int fIndex, int binNo) {
    return bf(fIndex, binNo).isActive();
  }


  public DynamicRow[] allRows() {
    return rows;
  }


}