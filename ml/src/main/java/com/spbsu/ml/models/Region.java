package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.List;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class Region extends Func.Stub implements BinOptimizedModel {
  private final BFGrid.BinaryFeature[] features;
  private final boolean[] mask;
  private final double inside;
  private final double outside;
  private final int maxFailed;
  private final int basedOn;
  private final double score;
  private final BFGrid grid;

  public BFGrid.BinaryFeature[] getFeatures() {
    return features.clone();
  }

  public boolean[] getMask() {
    return mask.clone();
  }

  public Region(final List<BFGrid.BinaryFeature> conditions, boolean[] mask, double inside, int basedOn, double score) {
    this(conditions, mask, inside, 0, basedOn, score, 0);
  }

  public Region(final List<BFGrid.BinaryFeature> conditions, boolean[] mask, double inside, double outside, int basedOn, double score, int maxFailed) {
    grid = conditions.get(0).row().grid();
    this.basedOn = basedOn;
    this.score = score;
    this.features = conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]);
    this.mask = mask;
    this.inside = inside;
    this.outside = outside;
    this.maxFailed = maxFailed;
  }

  @Override
  public double value(BinarizedDataSet bds, int pindex) {
    int failed = 0;
    for (int i = 0; i < features.length; i++) {
      if (bds.bins(features[i].findex)[pindex] > features[i].binNo != mask[i])
        ++failed;
    }
    return failed <= maxFailed ? inside : outside;
  }

  @Override
  public double value(Vec x) {
    int failed = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        ++failed;
    }
    if (failed > maxFailed) {
      return outside;
    }
    return inside;
  }

  @Override
  public int dim() {
    return grid.rows();
  }

  public boolean contains(Vec x) {
    int failed = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        ++failed;
    }
    return failed > maxFailed;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(maxFailed).append(":");
    builder.append(inside).append("/").append(outside).append("/").append(basedOn);
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
