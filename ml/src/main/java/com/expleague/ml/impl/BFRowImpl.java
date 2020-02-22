package com.expleague.ml.impl;

import com.expleague.ml.BFGrid;
import com.expleague.ml.meta.FeatureMeta;

import java.util.Arrays;

public class BFRowImpl implements BFGrid.Row {
  private BFGridImpl owner;
  public final int bfStart;
  public final int bfEnd;
  public final int origFIndex;
  public final double[] borders;
  public final BinaryFeatureImpl[] bfs;
  public final boolean isOneHot;

  public BFRowImpl(final BFGridImpl owner, final int bfStart, final int origFIndex, final double[] borders) {
    this(owner,bfStart,origFIndex,borders,new int[borders.length], false);
  }
  // TODO: why do we need bsStart?
  public BFRowImpl(final BFGridImpl owner, final int bfStart, final int origFIndex, final double[] borders, final int[] sizes, final boolean isOneHot) {
    this.owner = owner;
    this.bfStart = bfStart;
    this.bfEnd = bfStart + borders.length;
    this.origFIndex = origFIndex;
    this.borders = borders;
    bfs = new BinaryFeatureImpl[borders.length];
    for (int i = 0; i < borders.length; i++) {
      bfs[i] = new BinaryFeatureImpl(this, bfStart + i, origFIndex, i, borders[i], sizes[i], isOneHot);
    }
    this.isOneHot = isOneHot;
  }



  public BFRowImpl(final int bfStart, final int origFIndex, final double[] borders) {
    this(null, bfStart, origFIndex, borders);
  }

  public BFRowImpl(final int bfStart, final int origFIndex, final double[] borders, final int[] sizes) {
    this(null, bfStart, origFIndex, borders,sizes, false);
  }

  @Override
  public int bin(final double val) {
    int index = 0;
//      final int index = Arrays.binarySearch(borders, val);
//      return bfStart + (index >= 0 ? index : -index-1);
    while (index < borders.length && val > borders[index])
      index++;

    return index;
  }

  @Override
  public BinaryFeatureImpl bf(final int index) {
    return bfs[index];
  }

  @Override
  public double condition(final int border) {
    return borders[border];
  }

  @Override
  public int findex() {
    return origFIndex;
  }

  @Override
  public boolean ordered() {
    return !isOneHot;
  }

  @Override
  public int start() {
    return bfStart;
  }

  @Override
  public int end() {
    return bfEnd;
  }

  @Override
  public int size() {
    return bfEnd - bfStart;
  }

  @Override
  public boolean empty() {
    return bfEnd == bfStart;
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (!(o instanceof BFRowImpl)) return false;

    final BFRowImpl bfRow = (BFRowImpl) o;

    return bfStart == bfRow.bfStart && origFIndex == bfRow.origFIndex && Arrays.equals(borders, bfRow.borders);

  }

  @Override
  public int hashCode() {
    int result = bfStart;
    result = 31 * result + origFIndex;
    result = 31 * result + Arrays.hashCode(borders);
    return result;
  }

  @Override
  public BFGrid grid() {
    return owner;
  }

  @Override
  public FeatureMeta fmeta() {
    return owner.meta(findex());
  }

  void setOwner(final BFGridImpl owner) {
    this.owner = owner;
  }
}
