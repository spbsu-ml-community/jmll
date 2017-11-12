package com.expleague.ml.data.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.distributions.RandomVec;

public class PointEstimateBinarizedFeature extends BinarizedFeature.Stub implements BinarizedFeature {
  final byte[] byteStorage;
  final short[] shortStorage;

  public PointEstimateBinarizedFeature(final FeatureBinarization binarization,
                                       final RandomVec feature) {
    super(binarization, feature);
    final Vec values = feature.expectation();
    final int docCount = values.dim();
    final double[] borders = binarization.borders();
    if (borders.length > 254) {
      shortStorage = new short[docCount];
      byteStorage = null;
    }
    else {
      byteStorage = new byte[docCount];
      shortStorage = null;
    }

    for (int i = 0; i < docCount; ++i) {
      final int bin = binarization.bin(values.get(i));
      if (shortStorage != null) {
        shortStorage[i] = (short) bin;
      }
      else {
        byteStorage[i] = (byte) bin;
      }
    }
  }


  @Override
  public void visit(final int[] indices,
                    final FeatureVisitor visitor) {

    if (byteStorage != null) {
        for (int i = 0; i < (indices.length / 4) * 4; i += 4) {
          final int idx1 = indices[i];
          final int idx2 = indices[i+1];
          final int idx3 = indices[i+2];
          final int idx4 = indices[i+3];
          final byte bin1 = byteStorage[idx1];
          final byte bin2 = byteStorage[idx2];
          final byte bin3 = byteStorage[idx3];
          final byte bin4 = byteStorage[idx4];

          visitor.accept(idx1, i, bin1, 1.0);
          visitor.accept(idx2, i+1, bin2, 1.0);
          visitor.accept(idx3, i+2, bin3, 1.0);
          visitor.accept(idx4, i+3, bin4, 1.0);
        }

        for (int i = (indices.length / 4) * 4; i < indices.length; ++i) {
          final int idx = indices[i];
          final byte bin = byteStorage[idx];
          visitor.accept(idx, i, bin, 1.0);
        }
    }
    else if (shortStorage != null) {
      for (int i = 0; i < indices.length; ++i) {
        final int idx = indices[i];
        final int bin = shortStorage[idx];
        visitor.accept(idx, i, bin, 1.0);
      }
    }
    else {
      throw new RuntimeException("Error: empty storage");
    }
  }
}
