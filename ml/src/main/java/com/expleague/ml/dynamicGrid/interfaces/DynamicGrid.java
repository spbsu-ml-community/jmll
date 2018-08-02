package com.expleague.ml.dynamicGrid.interfaces;

import com.expleague.commons.math.vectors.Vec;

/**
 * Created by noxoomo on 23/07/14.
 */
public interface DynamicGrid {
  int rows();

  void setKnown(int hash);

  boolean isKnown(int hash);

  boolean isActive(int fIndex, int binNo);

  DynamicRow row(int feature);

  void binarize(Vec x, short[] folds);

  BinaryFeature bf(int fIndex, int binNo);

  DynamicRow nonEmptyRow();

  boolean addSplit(int feature);

  int[] hist();
}
