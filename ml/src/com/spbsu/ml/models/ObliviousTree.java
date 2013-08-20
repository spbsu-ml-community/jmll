package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

/**
* User: solar
* Date: 29.11.12
* Time: 5:35
*/
public class ObliviousTree extends Model {
  private final BFGrid.BinaryFeature[] features;
  private final double[] values;
  private final double[] basedOn;

  public ObliviousTree(final List<BFGrid.BinaryFeature> features, double[] values, double[] basedOn) {
    assert values.length == 1 << features.size();
    this.basedOn = basedOn;
    this.features = features.toArray(new BFGrid.BinaryFeature[features.size()]);
    this.values = values;
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
             .append(features[i]).append("@").append(basedOn[i]);
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
  public List<BFGrid.BinaryFeature> getFeatures(){
    List<BFGrid.BinaryFeature> ret = new ArrayList<BFGrid.BinaryFeature>();
    for(int i = 0;i < features.length;i++)
        ret.add(features[i]);
    return ret;
  }

  public double[] values() {
    return values;
  }

  public double[] based() {
    return basedOn;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof ObliviousTree)) return false;

    ObliviousTree that = (ObliviousTree) o;

    if (!Arrays.equals(features, that.features)) return false;
    if (!Arrays.equals(values, that.values)) return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result = Arrays.hashCode(features);
    result = 31 * result + Arrays.hashCode(values);
    result = 31 * result + Arrays.hashCode(basedOn);
    return result;
  }

  public BFGrid grid() {
    return features[0].row().grid();
  }
}
