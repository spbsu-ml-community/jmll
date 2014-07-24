package com.spbsu.ml.DynamicGrid.Models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.DynamicGrid.Interface.DynamicGrid;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.impl.BinarizedDynamicDataSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ObliviousTreeDynamicBin extends Func.Stub implements BinDynamicOptimizedModel{
  private final BinaryFeature[] features;
  private final double[] values;
  private final DynamicGrid grid;

  public ObliviousTreeDynamicBin(final List<BinaryFeature> features, double[] values) {
    grid = features.get(0).row().grid();
    this.features = features.toArray(new BinaryFeature[features.size()]);
    this.values = values;
  }

  @Override
  public int dim() {
    return grid.rows();
  }

  @Override
  public double value(Vec x) {
    int index = bin(x);
    return values[index];
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(values.length);
    builder.append("->(");
    for (int i = 0; i < features.length; i++) {
      builder.append(i > 0 ? ", " : "")
              .append(features[i]).append("@");
    }
    builder.append(")");
    builder.append("+[");
    for (double feature : values) {
      builder.append(feature).append(", ");
    }
    builder.delete(builder.length() - 2, builder.length());
    builder.append("]");
    return builder.toString();
  }

  public int bin(Vec x) {
    int index = 0;
    for (int i = 0; i < features.length; i++) {
      index <<= 1;
      if (features[i].value(x))
        index++;
    }
    return index;
  }

  public List<BinaryFeature> features() {
    List<BinaryFeature> ret = new ArrayList<>();
    for (int i = 0; i < features.length; i++)
      ret.add(features[i]);
    return ret;
  }

  public double[] values() {
    return values;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof ObliviousTreeDynamicBin)) return false;

    ObliviousTreeDynamicBin that = (ObliviousTreeDynamicBin) o;

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
    return features[0].row().grid();
  }

  @Override
  public double value(BinarizedDynamicDataSet bds, int pindex) {
    int index = 0;
    for (int i = 0; i < features.length; i++) {
      index <<= 1;
      if (bds.bins(features[i].fIndex())[pindex] > features[i].binNo())
        index++;
    }
    return values[index];
  }
}
