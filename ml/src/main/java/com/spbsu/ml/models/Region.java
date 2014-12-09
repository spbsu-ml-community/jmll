package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class Region extends RegionBase {
  private final BFGrid.BinaryFeature[] features;
  private final boolean[] mask;
  public final int maxFailed;
  public final int basedOn;
  public final double score;

  public BFGrid.BinaryFeature[] features() {
    return features.clone();
  }

  public boolean[] masks() {
    return mask.clone();
  }

  public Region(final List<BFGrid.BinaryFeature> conditions, boolean[] mask, double inside, int basedOn, double score) {
    this(conditions, mask, inside, 0, basedOn, score, 0);
  }

  public Region(final List<BFGrid.BinaryFeature> conditions, boolean[] mask, double inside, double outside, int basedOn, double score, int maxFailed) {
    super(conditions.get(0).row().grid(), inside, outside);
    this.basedOn = basedOn;
    this.score = score;
    this.features = conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]);
    this.mask = mask;
    this.maxFailed = maxFailed;
  }

  @Override
  public boolean contains(BinarizedDataSet bds, int pindex) {
    int failed = 0;
    for (int i = 0; i < features.length; i++) {
      if (bds.bins(features[i].findex)[pindex] > features[i].binNo != mask[i]) {
        if (++failed > maxFailed)
          return false;
      }
    }

    return true;
  }

  @Override
  public boolean contains(Vec x) {
    int failed = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        ++failed;
    }
    return failed <= maxFailed;
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

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof Region)) return false;
    Region that = (Region) o;
    return Arrays.equals(features, that.features) && Arrays.equals(mask, that.mask) && this.inside == that.inside && this.outside == that.outside && this.maxFailed == that.maxFailed && this.score == that.score && this.basedOn == that.basedOn;
  }

  public double score() {
    return score;
  }
}
