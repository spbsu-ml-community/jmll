package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;

import java.util.List;

/**
* User: solar
* Date: 29.11.12
* Time: 5:35
*/
public class ObliviousTree implements Model {
  private final BFGrid.BinaryFeature[] features;
  private final double[] values;
  private final double[] basedOn;
  private final double score;

  public ObliviousTree(final List<BFGrid.BinaryFeature> features, double[] values, double[] basedOn, double bestScore) {
    assert values.length == 1 << features.size();
    this.basedOn = basedOn;
    this.features = features.toArray(new BFGrid.BinaryFeature[features.size()]);
    this.values = values;
    this.score = bestScore;
  }

  @Override
  public double value(Vec x) {
    int index = bin(x);
    return values[index];
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(values.length).append("/").append(basedOn);
    builder.append(" ->(");
    for (int i = 0; i < features.length; i++) {
      builder.append(i > 0 ? ", " : "")
             .append(features[i]);
    }
    builder.append(")");
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
}
