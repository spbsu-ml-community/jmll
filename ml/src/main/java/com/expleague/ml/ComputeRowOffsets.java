package com.expleague.ml;

import com.expleague.commons.func.Computable;
import com.expleague.ml.data.BinarizedFeatureDataSet;
import com.expleague.ml.data.impl.BinarizedFeature;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.randomnessAware.ProcessRandomnessPolicy;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;
import gnu.trove.list.array.TIntArrayList;

/**
 * noxoomo
 */
public class ComputeRowOffsets implements Computable<BinarizedFeatureDataSet, ComputeRowOffsets> {
  private BinarizedFeatureDataSet dataSet;
  private int[] offsets;
  private int[] binOffsets;

  //offsets for all features + total bin feature count
  public int[] offsets() {
    if (offsets == null) {
      int offset = 0;
      TIntArrayList offsetsList = new TIntArrayList();

      for (final BinarizedFeature feature : dataSet.features()) {
        offsetsList.add(offset);
        offset += feature.binarization().features().length;
      }
      offsetsList.add(offset);
      synchronized (this) {
        offsets = offsetsList.toArray();
      }
    }
    return offsets;
  }

  public int[] binOffsets() {
    if (binOffsets == null) {
      int offset = 0;
      TIntArrayList offsetsList = new TIntArrayList();

      for (final BinarizedFeature feature : dataSet.features()) {
        offsetsList.add(offset);
        offset += feature.binarization().features().length + 1;
      }
      offsetsList.add(offset);
      synchronized (this) {
        binOffsets = offsetsList.toArray();
      }
    }
    return binOffsets;
  }

  @Override
  public ComputeRowOffsets compute(final BinarizedFeatureDataSet dataSet) {
    this.dataSet = dataSet;
    return this;
  }
}
