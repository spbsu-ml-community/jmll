package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.ArrayList;
import java.util.List;

/**
* User: solar
* Date: 29.11.12
* Time: 5:35
*/
public class Region extends Func.Stub implements BinOptimizedModel {
  private final BFGrid.BinaryFeature[] features;
  private final boolean[] mask;
  private final double value;
  private final int basedOn;
  private final double score;
  private final BFGrid grid;

  public BFGrid.BinaryFeature[] getFeatures() {
    return features.clone();
  }

  public boolean[] getMask() {
    return mask.clone();
  }

  public Region(final List<BFGrid.BinaryFeature> conditions, boolean[] mask, double value, int basedOn, double score) {
    grid = conditions.get(0).row().grid();
    this.basedOn = basedOn;
    this.score = score;
    this.features = conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]);
    this.mask = mask;
    this.value = value;
  }

  @Override
  public double value(BinarizedDataSet bds, int pindex) {
    for (int i = 0; i < features.length; i++) {
      if (bds.bins(features[i].findex)[pindex] > features[i].binNo ^ !mask[i])
        return 0;
    }
    return value;
  }

  @Override
  public double value(Vec x) {
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        return 0.;
    }
    return value;
  }

  @Override
  public int dim() {
    return grid.rows();
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

  public double score() {
    return score;
  }
}
