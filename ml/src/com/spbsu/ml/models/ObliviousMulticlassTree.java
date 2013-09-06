package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;

import java.util.Arrays;
import java.util.List;

/**
* User: solar
* Date: 29.11.12
* Time: 5:35
*/
public class ObliviousMulticlassTree extends ObliviousTree {
  private final boolean[][] masks;

  public ObliviousMulticlassTree(final List<BFGrid.BinaryFeature> features, double[] values, double[] basedOn, boolean[][] masks) {
    super(features, values, basedOn);
    this.masks = masks;
  }

  public double value(Vec x, int classNo) {
    final int bin = bin(x);
    final double v = values()[bin];
    return masks[bin][classNo] ? v : -v;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(super.toString());
    builder.append('<');
    for (int i = 0; i < masks.length; i++) {
      boolean[] mask = masks[i];
      if (i > 0)
        builder.append(", ");
      for (int j = 0; j < mask.length; j++) {
        builder.append(mask[j] ? 1 : 0);
      }
    }
    builder.append('>');
    return builder.toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof ObliviousMulticlassTree)) return false;
    return super.equals(o) && Arrays.equals(masks, ((ObliviousMulticlassTree)o).masks);
  }

  @Override
  public int hashCode() {
    return 31 * super.hashCode() + Arrays.hashCode(masks);
  }
}
