package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.Arrays;
import java.util.List;

/**
 * User: noxoomo
 */

public class LinearRegion extends  BinOptimizedModel.Stub  {
  private final BFGrid.BinaryFeature[] features;
  private final boolean[] mask;
  private final double bias;
  private final double[] values;
  private final BFGrid grid;


  public BFGrid.BinaryFeature[] features() {
    return features.clone();
  }

  public boolean[] masks() {
    return mask.clone();
  }

  public LinearRegion(final List<BFGrid.BinaryFeature> conditions,
                      final boolean[] mask,
                      final double bias,
                      final double[] values) {
    this.grid = conditions.size() > 0 ? conditions.get(0).row().grid() : null;
    this.features = conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]);
    this.mask = mask;
    this.bias = bias;
    this.values = values;
  }

  @Override
  public double value(final BinarizedDataSet bds,
                      final int pindex) {
    double result = bias;

    for (int i = 0; i < features.length; i++) {
      if (bds.bins(features[i].findex)[pindex] > features[i].binNo != mask[i]) {
        break;
      } else {
        result += values[i];
      }
    }

    return result;
  }
//
  @Override
  public  double value(final Vec x) {
    double result = bias;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i]) {
        break;
      }
      result += values[i];
    }
    return result;
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
//    builder.append(maxFailed).append(":");
//    builder.append(inside).append("/").append(outside).append("/").append(basedOn);
    builder.append(" ->");
    for (int i = 0; i < features.length; i++) {
      builder.append(" ")
              .append(features[i].findex)
              .append(mask[i] ? ">" : "<=")
              .append(features[i].condition);
    }
    builder.append("values: [");
    for (int i = 0; i < values.length; i++) {
      builder.append(values[i]).append(";");
    }
    builder.append("]");

    return builder.toString();
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof LinearRegion)) return false;
    final LinearRegion that = (LinearRegion) o;
    return Arrays.equals(features, that.features)
            && Arrays.equals(mask, that.mask) &&
            Arrays.equals(values, that.values);
  }

  @Override
  public int dim() {
    return grid.rows();
  }
}
