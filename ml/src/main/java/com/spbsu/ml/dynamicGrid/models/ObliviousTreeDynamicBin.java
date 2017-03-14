package com.spbsu.ml.dynamicGrid.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.dynamicGrid.impl.BinarizedDynamicDataSet;
import com.spbsu.ml.dynamicGrid.interfaces.BinaryFeature;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;

import java.util.Arrays;
import java.util.List;

public class ObliviousTreeDynamicBin extends Func.Stub implements BinDynamicOptimizedModel {
  private final BinaryFeature[] features;
  private final double[] values;
  private final DynamicGrid grid;

  public int depth() {
    return features.length;
  }


  public ObliviousTreeDynamicBin(final List<BinaryFeature> features, final double[] values) {
    this.grid = features.get(0).row().grid();
    this.features = features.toArray(new BinaryFeature[features.size()]);
    this.values = values;
  }

  @Override
  public int dim() {
    return grid.rows();
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
    for (int i = 0; i < features.length; i++) {
      builder.append(i > 0 ? ", " : "")
              .append(features[i]).append("@");
    }
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

  public BinaryFeature[] features() {
    return features;
  }

  public double[] values() {
    return values;
  }


  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof ObliviousTreeDynamicBin)) return false;

    final ObliviousTreeDynamicBin that = (ObliviousTreeDynamicBin) o;

    if (!Arrays.equals(features, that.features)) return false;
    if (!Arrays.equals(values, that.values)) return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(features);
    result = 31 * result + Arrays.hashCode(values);
    return result;
  }

  public DynamicGrid grid() {
    return grid;
  }

  @Override
  public double value(final BinarizedDynamicDataSet bds, final int pindex) {
    int index = 0;
    for (int i = 0; i < features.length; i++) {
      index <<= 1;
      if (bds.bins(features[i].fIndex())[pindex] > features[i].binNo())
        index++;
    }
    return values[index];
  }
}
