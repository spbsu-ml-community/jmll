package com.expleague.ml.data.tools.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.tools.FeatureSet;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.FeatureMeta;

import java.util.stream.IntStream;
import java.util.stream.Stream;

public class JoinedFeatureSet<T extends DSItem> extends FeatureSet.Stub<T> {
  private final int dim;
  private final int[] index;
  private final int[] fsStart;
  private final FeatureSet<? super T>[] fs;

  @SafeVarargs
  public JoinedFeatureSet(FeatureSet<? super T>... fs) {
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
  public void accept(T item) {
    for (int i = 0; i < fs.length; i++) {
      fs[i].accept(item);
    }
  }

  @Override
  public Vec advanceTo(Vec to) {
    int index = 0;
    for (int i = 0; i < fs.length; i++) {
      final int finalIndex = index;
      final Vec vec = fs[i].advance();
      IntStream.range(0, fs[i].dim()).forEach(idx -> to.set(finalIndex + idx, vec.get(idx)));
      index += fs[i].dim();
    }
    return to;
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

  @Override
  public int index(FeatureMeta meta) {
    int index = 0;
    for (int i = 0; i < fs.length; i++) {
      final int currentIndex = fs[i].index(meta);
      if (currentIndex >= 0)
        return index + currentIndex;
      index += fs[i].dim();
    }
    return super.index(meta);
  }

  @Override
  public Stream<FeatureSet<? super T>> components() {
    return Stream.of(fs).flatMap(FeatureSet::components);
  }
}
