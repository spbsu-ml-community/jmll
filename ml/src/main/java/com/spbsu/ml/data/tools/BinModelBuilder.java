package com.spbsu.ml.data.tools;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.models.ObliviousTree;

/**
 * Created by noxoomo on 31/07/14.
 */
public class BinModelBuilder {
  private final FullMatrixClassifierInfo result;
  private final int[] rowStarts;

  public BinModelBuilder(final BFGrid grid) {
    rowStarts = new int[grid.rows()];
    int gridSize = 0;
    for (int i = 0; i < grid.rows(); ++i)
      gridSize += grid.row(i).size();
    result = new FullMatrixClassifierInfo(gridSize);
    int currentIndex = 0;
    for (int rowIndex = 0; rowIndex < grid.rows(); ++rowIndex) {
      rowStarts[rowIndex] = currentIndex;
      final BFGrid.BFRow row = grid.row(rowIndex);
      for (int bin = 0; bin < row.size(); ++bin) {
        final BFGrid.BinaryFeature bf = row.bf(bin);
        result.binFeatures[currentIndex] = new BinaryFeatureStat(rowIndex, bf.condition);
        ++currentIndex;
      }
    }
  }


  public void append(final ObliviousTree tree, final Double weight) {
    final int treeDepth = tree.features().size();
    final int[] conditions = new int[treeDepth];
    final double[][] values;

    final BFGrid.BinaryFeature[] features = (BFGrid.BinaryFeature[]) tree.features().toArray();
    for (int i = 0; i < treeDepth; ++i) {
      final int depth = treeDepth - i - 1;
      conditions[i] = rowStarts[features[depth].findex] + features[depth].binNo;
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


