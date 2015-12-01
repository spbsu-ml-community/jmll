package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Trans;

import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class ObliviousMultiClassTree extends Trans.Stub {
  private final ObliviousTree binaryClassifier;
  private final boolean[][] masks;

  public ObliviousMultiClassTree(final List<BFGrid.BinaryFeature> features, final double[] values, final double[] basedOn, final boolean[][] masks) {
    binaryClassifier = new ObliviousTree(features, values, basedOn);
    this.masks = masks;
  }

  @Override
  public int ydim() {
    return masks.length;
  }

  @Override
  public int xdim() {
    return binaryClassifier.xdim();
  }

  public double value(final Vec x, final int classNo) {
    final int bin = binaryClassifier.bin(x);
    final double v = binaryClassifier.values()[bin];
    return masks[bin][classNo] ? v : -v;
  }

  @Override
  public Vec trans(final Vec x) {
    final Vec result = new ArrayVec(ydim());
    for (int c = 0; c < ydim(); c++) {
      result.set(c, value(x, c));
    }
    return result;
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    builder.append(binaryClassifier.toString());
    builder.append('<');
    for (int i = 0; i < masks.length; i++) {
      final boolean[] mask = masks[i];
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
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof ObliviousMultiClassTree)) return false;
    return binaryClassifier.equals(((ObliviousMultiClassTree) o).binaryClassifier) && Arrays.equals(masks, ((ObliviousMultiClassTree) o).masks);
  }

  @Override
  public int hashCode() {
    return 31 * binaryClassifier.hashCode() + Arrays.hashCode(masks);
  }

  public ObliviousTree binaryClassifier() {
    return binaryClassifier;
  }

  public boolean[] mask(final int i) {
    return masks[i];
  }
}
