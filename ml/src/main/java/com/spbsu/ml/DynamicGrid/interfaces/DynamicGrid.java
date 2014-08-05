package com.spbsu.ml.dynamicGrid.interfaces;

import com.spbsu.commons.math.vectors.Vec;

/**
 * Created by noxoomo on 23/07/14.
 */
public interface DynamicGrid {
  public int rows();

  public void setKnown(int hash);

  public boolean isKnown(int hash);

  public boolean isActive(int fIndex, int binNo);

  public DynamicRow row(int feature);

  public void binarize(Vec x, short[] folds);

  public BinaryFeature bf(int fIndex, int binNo);

  public DynamicRow nonEmptyRow();

  public boolean addSplit(int feature);

  public int[] hist();
}
