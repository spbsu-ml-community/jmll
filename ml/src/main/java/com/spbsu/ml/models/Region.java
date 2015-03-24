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
  public final BFGrid.BinaryFeature[] features;
  public final boolean[] mask;
  public final int maxFailed;
  public final int basedOn;
  public final double score;

  public BFGrid.BinaryFeature[] features() {
    return features.clone();
  }

  public boolean[] masks() {
    return mask.clone();
  }

  public Region(final List<BFGrid.BinaryFeature> conditions, final boolean[] mask, final double inside, final int basedOn, final double score) {
    this(conditions, mask, inside, 0, basedOn, score, 0);
  }

  public Region(final List<BFGrid.BinaryFeature> conditions, final boolean[] mask, final double inside, final double outside, final int basedOn, final double score, final int maxFailed) {
    super(conditions.size() > 0 ? conditions.get(0).row().grid() : null, inside, outside);
    this.basedOn = basedOn;
    this.score = score;
    this.features = conditions.toArray(new BFGrid.BinaryFeature[conditions.size()]);
    this.mask = mask;
    this.maxFailed = maxFailed;
  }


  public Region(final Region base, final double inside, final double outside) {
    super(base.grid, inside, outside);
    this.basedOn = base.basedOn;
    this.score = base.score();
    this.features = base.features;
    this.mask = base.mask;
    this.maxFailed = base.maxFailed;
  }

  @Override
  public boolean contains(final BinarizedDataSet bds, final int pindex) {
    int failed = 0;
    for (int i = 0; i < features.length; i++) {
      if (bds.bins(features[i].findex)[pindex] > features[i].binNo != mask[i]) {
        if (++failed > maxFailed) {
          return false;
        }
      }
    }

    return true;
  }

  @Override
  public boolean contains(final Vec x) {
    int failed = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        ++failed;
    }
    return failed <= maxFailed;
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
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
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof Region)) return false;
    final Region that = (Region) o;
    return Arrays.equals(features, that.features) && Arrays.equals(mask, that.mask) && this.inside == that.inside && this.outside == that.outside && this.maxFailed == that.maxFailed && this.score == that.score && this.basedOn == that.basedOn;
  }

  public double score() {
    return score;
  }
}
