package com.spbsu.ml.dynamicGrid.impl;

import com.spbsu.ml.dynamicGrid.interfaces.BinaryFeature;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicRow;

/**
 * Created by noxoomo on 23/07/14.
 */
public class StaticRow implements DynamicRow {
  private final int origFIndex;
  private DynamicGrid grid = null;
  private final BinaryFeature[] bfs;

  public StaticRow(DynamicGrid grid, int origFIndex, double[] borders) {
    this.origFIndex = origFIndex;
    this.grid = grid;
    bfs = new BinaryFeature[borders.length];
    for (int i = 0; i < borders.length; ++i) {
      bfs[i] = new BinaryFeatureImpl(this, origFIndex, borders[i], i);
      bfs[i].setActive(true);
    }
  }

  @Override
  public int origFIndex() {
    return origFIndex;
  }

  @Override
  public int size() {
    return bfs.length;
  }

  @Override
  public DynamicGrid grid() {
    return grid;
  }

  @Override
  public boolean addSplit() {
    return false;
  }

  @Override
  public boolean empty() {
    return size() == 0;
  }

  @Override
  public BinaryFeature bf(int binNo) {
    return bfs[binNo];
  }

  @Override
  public void setOwner(DynamicGrid grid) {
    this.grid = grid;
  }

  @Override
  public int bin(double value) {
    int index = 0;
    while (index < size() && value > bfs[index].condition())
      index++;
    return index;
  }
}
