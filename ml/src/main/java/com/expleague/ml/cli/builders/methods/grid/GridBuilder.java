package com.expleague.ml.cli.builders.methods.grid;

import com.expleague.commons.func.Factory;
import com.expleague.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.GridTools;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.BFGrid;
import com.expleague.ml.impl.BFRowImpl;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TIntHashSet;

import java.util.HashSet;
import java.util.Set;


public class GridBuilder implements Factory<BFGrid> {
  private BFGrid cooked;
  private VecDataSet ds;
  private Set<Integer> catFeatureIds = new HashSet<>();
  private Set<Integer> ignoredColumns = new HashSet<>();
  private final TIntHashSet known = new TIntHashSet();
  private int binsCount = 32;
  private int bfCount = 0;
  private BFRowImpl[] rows;
  private int oneHotLimit = 255;
  //temp
  private double[] feature;

  public GridBuilder() {
  }

  public void setGrid(final BFGrid cooked) {
    this.cooked = cooked;
  }

  public void setBinsCount(final int binsCount) {
    this.binsCount = binsCount;
  }

  public void setDataSet(final VecDataSet dataSet) {
    this.rows = new BFRowImpl[dataSet.xdim()];
    this.ds = dataSet;
    this.feature = new double[dataSet.length()];
  }

  public GridBuilder addCatFeatureIds(final Set<Integer> catFeatureIds) {
    this.catFeatureIds.addAll(catFeatureIds);
    return this;
  }

  public GridBuilder addIgnoredColumns(final Set<Integer> ignoredColumns) {
    this.ignoredColumns.addAll(ignoredColumns);
    return this;
  }

  @Override
  public BFGrid create() {
    if (cooked == null) {
      cooked = build();
    }
    return cooked;
  }

  private BFGrid build() {
    if (ds == null) {
      throw new RuntimeException("Set dataset before build");
    }
    final int dim = ds.xdim();

    for (int f = 0; f < dim; f++) {
      if (ignoredColumns.contains(f)) {
        continue;
      }
      if (catFeatureIds.contains(f)) {
        addCatFeature(f);
      }
      else {
        addFloatFeature(f);
      }
    }
    return new BFGridImpl(rows);
  }

  private void addCatFeature(final int f) {
    final double[] fakeBorders = GridTools.sortUnique(ds.data().col(f));
    if (fakeBorders.length > oneHotLimit) {
      throw new RuntimeException("Error: we support <255 cat features levels currently");
    }

    final boolean haveDiffrentElements = fakeBorders.length > 1;
    if (!haveDiffrentElements) {
      return;
    }
    rows[f] = new BFRowImpl(null, bfCount, f, fakeBorders, new int[fakeBorders.length], true);
    bfCount += fakeBorders.length;
  }
  private void addFloatFeature(final int f) {
    final ArrayPermutation permutation = new ArrayPermutation(ds.order(f));
    final int[] order = permutation.direct();
    final int[] reverse = permutation.reverse();


    // fixme: code duplication, it is the same as GridTools.medianGrid
    boolean haveDifferentElements = false;
    for (int i = 1; i < order.length; i++)
      if (order[i] != order[0])
        haveDifferentElements = true;
    if (!haveDifferentElements) {
      return;
    }
    for (int i = 0; i < feature.length; i++)
      feature[i] = ds.at(order[i]).get(f);
    final TIntArrayList borders = GridTools.greedyLogSumBorders(feature, binsCount);
    final TDoubleArrayList dborders = new TDoubleArrayList();
    final TIntArrayList sizes = new TIntArrayList();
    { // drop existing
      // TODO: anyway I don't understand, why do we need crcs
      final int[] crcs = new int[borders.size()];
      for (int i = 0; i < ds.length(); i++) { // unordered index
        final int orderedIndex = reverse[i];
        for (int b = 0; b < borders.size() && orderedIndex >= borders.get(b); b++) {
          crcs[b] = (crcs[b] * 31) + (i + 1);
        }
      }
      // TODO: it's like we compute somewhat hash code to skip some borders but why?
      for (int b = 0; b < borders.size() - 1; b++) {
        if (known.contains(crcs[b]))
          continue;
        known.add(crcs[b]);
        dborders.add(feature[borders.get(b) - 1]);
        sizes.add(borders.get(b));
      }
    }
    // TODO: for what bfCount is used?
    rows[f] = new BFRowImpl(bfCount, f, dborders.toArray(), sizes.toArray());
    bfCount += dborders.size();
  }
}
