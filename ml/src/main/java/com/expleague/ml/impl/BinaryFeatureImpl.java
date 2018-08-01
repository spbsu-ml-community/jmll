package com.expleague.ml.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.impl.BinarizedDataSet;

public class BinaryFeatureImpl implements BFGrid.Feature {
  private final BFRowImpl bfRow;
  public final int bfIndex;
  public final int findex;
  public final int binNo;
  public final double condition;
  public final double size;
  public final boolean oneHot;

  public BinaryFeatureImpl(final BFRowImpl bfRow, final int bfIndex, final int findex, final int binNo, final double condition) {
    this(bfRow, bfIndex, findex, binNo, condition, 0, false);
  }

  public BinaryFeatureImpl(final BFRowImpl bfRow, final int bfIndex, final int findex,
                           final int binNo, final double condition, int size, final boolean oneHot) {
    this.bfRow = bfRow;
    this.bfIndex = bfIndex;
    this.findex = findex;
    this.binNo = binNo;
    this.condition = condition;
    this.size = size;
    this.oneHot = oneHot;
  }


  @Override
  public boolean value(final byte[] folds) {
    return value(folds[findex]);
  }

  @Override
  public boolean value(final byte fold) {
    return oneHot
        ? fold != binNo
        : fold > binNo;
  }

  @Override
  public boolean value(final Vec vec) {
    return oneHot ? vec.get(findex) != condition
                  : vec.get(findex) > condition;
  }

  @Override
  public boolean value(int index, BinarizedDataSet bds) {
    return value(bds.bins(findex)[index]);
  }

  @Override
  public int findex() {
    return findex;
  }

  @Override
  public int bin() {
    return binNo;
  }

  @Override
  public int index() {
    return bfIndex;
  }

  @Override
  public double condition() {
    return condition;
  }

  @Override
  public double power() {
    return size;
  }

  @Override
  public BFRowImpl row() {
    return bfRow;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof BinaryFeatureImpl)) return false;

    final BinaryFeatureImpl that = (BinaryFeatureImpl) o;

    return bfIndex == that.bfIndex && bfRow.equals(that.bfRow);

  }

  @Override
  public int hashCode() {
    int result = bfRow.hashCode();
    result = 31 * result + bfIndex;
    return result;
  }

  @Override
  public String toString() {
    return String.format(oneHot ? "f[%d] = %g" : "f[%d] > %g", findex, condition);
  }
}
