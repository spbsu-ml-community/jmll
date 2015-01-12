package com.spbsu.ml.data.tools;

import com.spbsu.ml.dynamicGrid.interfaces.BinaryFeature;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicRow;
import com.spbsu.ml.dynamicGrid.models.ObliviousTreeDynamicBin;

/**
 * Created by noxoomo on 31/07/14.
 */
public class DynamicBinModelBuilder {
  private final FullMatrixClassifierInfo result;
  private final int[] rowStarts;

  public DynamicBinModelBuilder(final DynamicGrid grid) {
    rowStarts = new int[grid.rows()];
    int gridSize = 0;
    for (int i = 0; i < grid.rows(); ++i)
      gridSize += grid.row(i).size();
    result = new FullMatrixClassifierInfo(gridSize);
    int currentIndex = 0;
    for (int rowIndex = 0; rowIndex < grid.rows(); ++rowIndex) {
      rowStarts[rowIndex] = currentIndex;
      final DynamicRow row = grid.row(rowIndex);
      for (int bin = 0; bin < row.size(); ++bin) {
        final BinaryFeature bf = row.bf(bin);
        result.binFeatures[currentIndex] = new BinaryFeatureStat(rowIndex, bf.condition());
        ++currentIndex;
      }
    }
  }


  public void append(final ObliviousTreeDynamicBin tree, final Double weight) {
    final int[] conditions = new int[tree.depth()];
    final double[][] values;

    final BinaryFeature[] features = tree.features();
    for (int i = 0; i < tree.depth(); ++i) {
      final int depth = tree.depth() - i - 1;
      conditions[i] = rowStarts[features[depth].fIndex()] + features[depth].binNo();
    }
    values = new double[1][1 << conditions.length];
    final double[] leaveValues = tree.values();

    for (int i = 0; i < (1 << conditions.length); ++i) {
      values[0][i] = leaveValues[i];
    }
    for (int i = 0; i < values[0].length; ++i) {
      values[0][i] *= weight;
    }
    result.trees.add(new TreeStat(conditions, values));
  }

  public FullMatrixClassifierInfo build() {
    return result;
  }
}


