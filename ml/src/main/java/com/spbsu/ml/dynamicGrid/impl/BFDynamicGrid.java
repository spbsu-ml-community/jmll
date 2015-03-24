package com.spbsu.ml.dynamicGrid.impl;

import com.spbsu.commons.func.Converter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.stats.OrderByFeature;
import com.spbsu.ml.dynamicGrid.interfaces.BinaryFeature;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicRow;
import com.spbsu.ml.io.DynamicGridStringConverter;
import gnu.trove.set.hash.TIntHashSet;

public class BFDynamicGrid implements DynamicGrid {
  private final DynamicRow[] rows;
  private final TIntHashSet known = new TIntHashSet();
  private final DynamicRow leastNonEmptyRow;


  public BFDynamicGrid(final VecDataSet ds, final int minSplits) {
    final OrderByFeature byFeature = ds.cache().cache(OrderByFeature.class, DataSet.class);

    final MedianRow[] rows = new MedianRow[ds.data().columns()];
    for (int f = 0; f < ds.data().columns(); ++f) {
      final ArrayPermutation permutation = byFeature.orderBy(f);
      final int[] order = permutation.direct();
      final double[] feature = new double[order.length];
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

  public BFDynamicGrid(final DynamicRow[] rows) {
    DynamicRow least = null;
    for (int f = 0; f < rows.length; ++f)
      if (!rows[f].empty()) {
        least = rows[f];
        break;
      }
    this.rows = rows;
    leastNonEmptyRow = least;
    for (final DynamicRow row : rows)
      row.setOwner(this);
  }


  @Override
  public DynamicRow row(final int feature) {
    return feature < rows.length ? rows[feature] : null;
  }

  @Override
  public void binarize(final Vec x, final short[] folds) {
    for (int i = 0; i < x.dim(); i++) {
      folds[i] = rows[i].bin(x.get(i));
    }

  }

  @Override
  public BinaryFeature bf(final int fIndex, final int binNo) {
    return rows[fIndex].bf(binNo);
  }

  @Override
  public DynamicRow nonEmptyRow() {
    return leastNonEmptyRow;
  }

  @Override
  public boolean addSplit(final int feature) {
    return rows[feature].addSplit();
  }


  @Override
  public int[] hist() {
    final int[] counts = new int[rows.length];
    for (int f = 0; f < rows.length; ++f) {
      counts[f] = rows[f].size();
    }
    return counts;
  }


  @Override
  public int rows() {
    return rows.length;
  }

  @Override
  public void setKnown(final int hash) {
    known.add(hash);
  }

  @Override
  public boolean isKnown(final int hash) {
    return known.contains(hash);
  }


  @Override
  public boolean isActive(final int fIndex, final int binNo) {
    return bf(fIndex, binNo).isActive();
  }


  public DynamicRow[] allRows() {
    return rows;
  }


  public static final Converter<DynamicGrid, CharSequence> CONVERTER = new DynamicGridStringConverter();

  @Override
  public String toString() {
    return CONVERTER.convertTo(this).toString();
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof DynamicGrid)) return false;

    final DynamicGrid grid = (DynamicGrid) o;

    if (this.rows() != grid.rows()) return false;

    for (int feature = 0; feature < rows(); ++feature) {
      final DynamicRow thisRow = this.row(feature);
      final DynamicRow otherRow = grid.row(feature);
      if (thisRow.size() != otherRow.size())
        return false;
      for (int bin = 0; bin < thisRow.size(); ++bin) {
        final BinaryFeature thisBF = thisRow.bf(bin);
        final BinaryFeature other = otherRow.bf(bin);
        if (Math.abs(thisBF.condition() - other.condition()) > 1e-9) return false;
        if (thisBF.fIndex() != other.fIndex()) return false;
      }
    }
    return true;
  }

}