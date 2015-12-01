package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinModelWithGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * User: noxoomo
 */

public class TransObliviousTree extends  BinOptimizedModel.Stub implements BinModelWithGrid {

  private final BFGrid.BinaryFeature[] features;
  private final Trans[] values;
  private final BFGrid grid;

  public TransObliviousTree(final List<BFGrid.BinaryFeature> features, final Trans[] values) {
    grid = features.get(0).row().grid();
    this.features = features.toArray(new BFGrid.BinaryFeature[features.size()]);
    this.values = values;
  }

  @Override
  public int dim() {
    return grid.rows();
  }

  @Override
  public double value(final Vec x) {
    final int index = bin(x);
    return values[index].trans(x).get(0);
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
    for (final Trans feature : values) {
      builder.append(feature.toString()).append(", ");
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

  public List<BFGrid.BinaryFeature> features() {
    final List<BFGrid.BinaryFeature> ret = new ArrayList<BFGrid.BinaryFeature>();
    Collections.addAll(ret, features);
    return ret;
  }

  public Trans[] values() {
    return values;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof TransObliviousTree)) return false;
    final TransObliviousTree that = (TransObliviousTree) o;
    return Arrays.equals(features, that.features) && Arrays.equals(values, that.values);
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(features);
    result = 31 * result + Arrays.hashCode(values);
    return result;
  }

  public BFGrid grid() {
    return features[0].row().grid();
  }

  @Override
  public double value(final BinarizedDataSet bds, final int pindex) {
    int index = 0;
    for (int i = 0; i < features.length; i++) {
      index <<= 1;
      if (bds.bins(features[i].findex)[pindex] > features[i].binNo)
        index++;
    }
    //dirty hack with cast
    return values[index].trans(((VecDataSet)bds.original()).data().row(pindex)).get(0);
  }

}
