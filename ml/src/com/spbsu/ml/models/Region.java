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
public class Region extends Model {
  private final BFGrid.BinaryFeature[] features;
  private final boolean[] mask;
  private final double value;
  private final int basedOn;

  public Region(final List<BFGrid.BinaryFeature> conditions, boolean[] mask, double value, int basedOn, double bestScore) {
    this.basedOn = basedOn;
    this.features = conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]);
    this.mask = mask;
    this.value = value;
  }

  @Override
  public double value(Vec x) {
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        return 0.;
    }
    return value;
  }

  public boolean contains(Vec x) {
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        return false;
    }

    return true;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(value).append("/").append(basedOn);
    builder.append(" ->");
    for (int i = 0; i < features.length; i++) {
      builder.append(" ")
             .append(features[i].findex)
             .append(mask[i] ? ">" : "<=")
             .append(features[i].condition);
    }
    return builder.toString();
  }
}
