package com.spbsu.ml.DynamicGrid.Impl;

import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
import com.spbsu.ml.DynamicGrid.Interface.DynamicRow;
import com.spbsu.ml.DynamicGrid.Trees.BFDynamicOptimizationSubset;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;

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
  private final int[][] bins;
  public final List<BinaryFeature> queue = new ArrayList<>();

  public BinarizedDynamicDataSet(DataSet base, DynamicGrid grid) {
    this.base = base;
    this.grid = grid;
    int xdim = ((VecDataSet) base).xdim();
    bins = new int[xdim][];
    for (int f = 0; f < bins.length; f++) {
      bins[f] = new int[base.length()];
    }
    int[] binarization = new int[grid.rows()];
    for (int t = 0; t < base.length(); t++) {
      grid.binarize(((VecDataSet) base).data().row(t), binarization);
      for (int f = 0; f < bins.length; f++) {
        bins[f][t] = binarization[f];
      }
    }
  }

  public boolean addSplit(int f) {
    if (grid.addSplit(f)) {
      updateBins(f);
      return true;
    }
    return false;
  }

  public void updateBins() {
    for (int f1 = 0; f1 < bins.length; f1++) {
      bins[f1] = new int[base.length()];
    }
    int[] binarization = new int[grid.rows()];
    for (int t = 0; t < base.length(); t++) {
      grid.binarize(((VecDataSet) base).data().row(t), binarization);
      for (int f1 = 0; f1 < bins.length; f1++) {
        bins[f1][t] = binarization[f1];
      }
    }
  }

  private void updateBins(int f) {
    DynamicRow row = grid.row(f);
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

  public int[] bins(int findex) {
    return bins[findex];
  }

  public void queueSplit(BinaryFeature bf) {
    queue.add(bf);
  }


  public boolean acceptQueue(List<BFDynamicOptimizationSubset> leaves) {
    if (queue.size() > 0) {
      int[] origFIndexes = new int[queue.size()];
      for (int i = 0; i < queue.size(); ++i) {
        origFIndexes[i] = queue.get(i).fIndex();
      }

      for (BinaryFeature feature : queue) {
        feature.row().addSplit();
        feature.setActive(true);
      }
      for (BinaryFeature feature : queue)
        updateBins(feature.fIndex());
      queue.clear();
//            java 8 parallel version
//            leaves.parallelStream().forEach((leave) -> leave.rebuild(origFIndexes));
      for (BFDynamicOptimizationSubset leave : leaves)
        leave.rebuild(origFIndexes);

      return true;
    }
    return false;

  }
}
