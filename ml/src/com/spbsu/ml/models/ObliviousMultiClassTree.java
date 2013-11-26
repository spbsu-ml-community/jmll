package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.VecFuncStub;

import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 29.11.12
 * Time: 5:35
 */
public class ObliviousMultiClassTree extends VecFuncStub {
  private final ObliviousTree binaryClassifier;
  private final boolean[][] masks;

  public ObliviousMultiClassTree(final List<BFGrid.BinaryFeature> features, double[] values, double[] basedOn, boolean[][] masks) {
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

  public double value(Vec x, int classNo) {
    final int bin = binaryClassifier.bin(x);
    final double v = binaryClassifier.values()[bin];
    return masks[bin][classNo] ? v : -v;
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append(binaryClassifier.toString());
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

  public boolean[] mask(int i) {
    return masks[i];
  }
}
