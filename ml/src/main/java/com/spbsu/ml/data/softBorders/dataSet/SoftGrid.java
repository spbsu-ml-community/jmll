package com.spbsu.ml.data.softBorders.dataSet;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by noxoomo on 14/12/2016.
 */
public class SoftGrid {
  private final List<SoftRow> rows;
  private SoftRow nonEmptyRow;
  private SoftRow.BinFeature[] binFeatures;

  public int rowsCount() {
    return rows.size();
  }

  public int binFeatureCount() {
    return binFeatures.length;
  }

  public SoftRow.BinFeature bf(final int idx) {
    return binFeatures[idx];
  }

  private SoftGrid() {
    this.rows = new ArrayList<>();
  }

  public SoftRow nonEmptyRow() {
    return nonEmptyRow;
  }

  public SoftRow row(final int featureIdx) {
    return rows.get(featureIdx);
  }

  public static class SoftRow {
    final WeightedFeature feature;
    final BinFeature[] binFeatures;
    final int featureIdx;
    final SoftGrid owner;
    final Vec binsDistributions;

    private SoftRow(final SoftGrid owner,
                    final int featureIdx,
                    final int featureOffset,
                    final WeightedFeature feature,
                    final double[] binsDistributions,
                    final double[] borders) {
      this.owner = owner;
      this.featureIdx = featureIdx;
      this.feature = feature;
      this.binsDistributions = new ArrayVec(binsDistributions);
      if (borders.length > 0) {
        assert (binsDistributions.length == (borders.length + 1) * feature.size());
      }
      this.binFeatures = new BinFeature[borders.length];
      for (int i = 0; i < borders.length; ++i) {
        binFeatures[i] = new BinFeature(this, featureOffset + i, i, borders[i]);
      }
    }

    public SoftGrid grid() {
      return owner;
    }

    public int binFeatureCount() {
      return binFeatures.length;
    }

    public int featureIdx() {
      return featureIdx;
    }

    public final Vec binsDistribution(final int rk) {
      return binsDistributions.sub(rk * (binFeatures.length + 1), binFeatures.length + 1);
    }

    public final int rank(final double value) {
      return feature.rankByValue(value);
    }

    public BinFeature binFeature(final int b) {
      return binFeatures[b];
    }

    public byte bin(final double v) {
      byte bin = 0;
      while (bin < binFeatures.length && v > binFeatures[bin].border) {
        ++bin;
      }
      return bin;
    }

    public static class BinFeature {
      final SoftRow owner;
      public final int bfIndex;
      public final int binIdx;
      public final double border;

      BinFeature(final SoftRow owner,
                 final int bfIndex,
                 final int binIdx,
                 final double border) {
        this.owner = owner;
        this.bfIndex = bfIndex;
        this.binIdx = binIdx;
        this.border = border;
      }

      public boolean value(Vec floatFeature) {
        return floatFeature.get(owner.featureIdx()) > border;
      }

      public SoftRow row() {
        return owner;
      }

      public SoftGrid grid() {
        return owner.grid();
      }

      @Override
      public String toString() {
        return String.format("f[%d] > %g",owner.featureIdx, border);
      }

    }
  }

  static Builder createBuilder() {
    return new Builder();
  }

  public static class Builder {
    final SoftGrid grid = new SoftGrid();
    List<SoftRow.BinFeature> binFeatures = new ArrayList<>();
    int featureIdx = 0;

    Builder addRow(final WeightedFeature feature,
                   final double[] binsDist,
                   final double[] borders) {

      final SoftRow nextRow = new SoftRow(grid, featureIdx++, binFeatures.size(), feature, binsDist, borders);
      grid.rows.add(nextRow);
      for (SoftRow.BinFeature binFeature : nextRow.binFeatures) {
        binFeatures.add(binFeature);
      }
      return this;
    }

    SoftGrid build() {
      for (SoftRow row : grid.rows) {
        if (row.binFeatureCount() > 0) {
          grid.nonEmptyRow = row;
          break;
        }
      }
      grid.binFeatures = binFeatures.toArray(new SoftRow.BinFeature[binFeatures.size()]);
      return grid;
    }
  }

}
