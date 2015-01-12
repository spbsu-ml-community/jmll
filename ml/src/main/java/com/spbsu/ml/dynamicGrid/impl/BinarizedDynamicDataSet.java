package com.spbsu.ml.dynamicGrid.impl;

import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.dynamicGrid.interfaces.BinaryFeature;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicRow;
import com.spbsu.ml.dynamicGrid.trees.BFDynamicOptimizationSubset;

import java.util.ArrayList;
import java.util.List;

/**
 * User: solar
 * Date: 05.12.12
 * Time: 21:19
 */
public class BinarizedDynamicDataSet {
  private final DataSet base;
  private final DynamicGrid grid;
  private final short[][] bins;
  public final List<BinaryFeature> queue = new ArrayList<>();

  public BinarizedDynamicDataSet(final DataSet base, final DynamicGrid grid) {
    this.base = base;
    this.grid = grid;
    final int xdim = ((VecDataSet) base).xdim();
    bins = new short[xdim][];
    for (int f = 0; f < bins.length; f++) {
      bins[f] = new short[base.length()];
    }
    final short[] binarization = new short[grid.rows()];
    for (int t = 0; t < base.length(); t++) {
      grid.binarize(((VecDataSet) base).data().row(t), binarization);
      for (int f = 0; f < bins.length; f++) {
        bins[f][t] = binarization[f];
      }
    }
  }

  public boolean addSplit(final int f) {
    if (grid.addSplit(f)) {
      updateBins(f);
      return true;
    }
    return false;
  }

  public void updateBins() {
    for (int f1 = 0; f1 < bins.length; f1++) {
      bins[f1] = new short[base.length()];
    }
    final short[] binarization = new short[grid.rows()];
    for (int t = 0; t < base.length(); t++) {
      grid.binarize(((VecDataSet) base).data().row(t), binarization);
      for (int f1 = 0; f1 < bins.length; f1++) {
        bins[f1][t] = binarization[f1];
      }
    }
  }

  private void updateBins(final int f) {
    final DynamicRow row = grid.row(f);
    for (int t = 0; t < bins[f].length; ++t) {
      bins[f][t] = row.bin(((VecDataSet) base).at(t).get(f));
    }
  }


  public DataSet original() {
    return base;
  }

  public DynamicGrid grid() {
    return grid;
  }

  public short[] bins(final int findex) {
    return bins[findex];
  }

  public void queueSplit(final BinaryFeature bf) {
    queue.add(bf);
  }


  public boolean acceptQueue(final List<BFDynamicOptimizationSubset> leaves) {
    if (queue.size() > 0) {
      final int[] origFIndexes = new int[queue.size()];
      for (int i = 0; i < queue.size(); ++i) {
        origFIndexes[i] = queue.get(i).fIndex();
      }

      for (final BinaryFeature feature : queue) {
        feature.row().addSplit();
        feature.setActive(true);
      }
      for (final BinaryFeature feature : queue)
        updateBins(feature.fIndex());
      queue.clear();
//            java 8 parallel version
//            leaves.parallelStream().forEach((leave) -> leave.rebuild(origFIndexes));
      for (final BFDynamicOptimizationSubset leave : leaves)
        leave.rebuild(origFIndexes);

      return true;
    }
    return false;

  }
}
