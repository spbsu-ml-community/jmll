package com.spbsu.ml.models;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.softBorders.dataSet.SoftGrid;

import java.util.Arrays;

/**
 * User: noxoomo
 */

public class SoftObliviousTree extends Func.Stub implements Func {
  final SoftGrid grid;
  private final SoftGrid.SoftRow.BinFeature[] features;
  private final double[] values;

  public SoftObliviousTree(final SoftGrid.SoftRow.BinFeature[] features,
                           final double[] values) {
    this.grid = features[0].grid();
    this.features = features;
    this.values = values;
  }

  @Override
  public int dim() {
    return grid.rowsCount();
  }

  @Override
  public double value(final Vec x) {
    final int index = bin(x);
    return values[index];
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(values.length);
    builder.append("->(");

    builder.append(")");
    builder.append("+[");
    for (final double feature : values) {
      builder.append(feature).append(", ");
    }
    builder.delete(builder.length() - 2, builder.length());
    builder.append("]");
    return builder.toString();
  }

  public int bin(final Vec x) {
    int index = 0;
    for (int i = 0; i < features.length; i++) {
      index <<= 1;
      if (features[i].value(x))
        index++;
    }
    return index;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof SoftObliviousTree)) return false;

    final SoftObliviousTree that = (SoftObliviousTree) o;

    if (grid != null ? !grid.equals(that.grid) : that.grid != null) return false;
    // Probably incorrect - comparing Object[] arrays with Arrays.equals
    if (!Arrays.equals(features, that.features)) return false;
    return Arrays.equals(values, that.values);

  }

  @Override
  public int hashCode() {
    int result = grid != null ? grid.hashCode() : 0;
    result = 31 * result + Arrays.hashCode(features);
    result = 31 * result + Arrays.hashCode(values);
    return result;
  }
}
