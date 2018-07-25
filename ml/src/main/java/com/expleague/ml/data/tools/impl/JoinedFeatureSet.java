package com.expleague.ml.data.tools.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.tools.FeatureSet;
import com.expleague.ml.meta.FeatureMeta;

import java.util.stream.IntStream;
import java.util.stream.Stream;

public class JoinedFeatureSet implements FeatureSet {
  private final int dim;
  private final int[] index;
  private final int[] fsStart;
  private final FeatureSet[] fs;

  public JoinedFeatureSet(FeatureSet... fs) {
    final int dim = Stream.of(fs).mapToInt(FeatureSet::dim).sum();
    final int[] index = new int[dim];
    final int[] fsStart = new int[fs.length];
    int idx = 0;
    for (int i = 0; i < fs.length; i++) {
      fsStart[i] = idx;
      for (int j = 0; j < fs[i].dim(); j++, idx++) {
        index[idx] = i;
      }
    }

    this.dim = dim;
    this.index = index;
    this.fsStart = fsStart;
    this.fs = fs;
  }

  @Override
  public Vec advance() {
    final Vec result = new ArrayVec(dim);
    int index = 0;
    for (int i = 0; i < fs.length; i++) {
      final int finalIndex = index;
      final Vec vec = fs[i].advance();
      IntStream.range(0, fs[i].dim()).forEach(idx -> result.set(finalIndex + idx, vec.get(idx)));
      index += fs[i].dim();
    }
    return result;
  }

  @Override
  public Vec advance(FeatureMeta... features) {
    return null;
  }

  @Override
  public int dim() {
    return dim;
  }

  @Override
  public FeatureMeta meta(int idx) {
    int fsIndex = index[idx];
    return fs[fsIndex].meta(idx - fsStart[fsIndex]);
  }
}
