package com.expleague.ml.loss.pairwise;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.IntSeqBuilder;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.GroupedDSItem;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;

import static java.lang.Math.*;

@SuppressWarnings("unused")
public class PairwiseLLLogit extends FuncC1.Stub implements TargetFunc {
  protected final Vec target;
  private final DataSet<? extends GroupedDSItem> owner;
  private final int[][] groups;

  public PairwiseLLLogit(final Vec target, final DataSet<? extends GroupedDSItem> owner) {
    this(target.stream(), owner);
  }

  public PairwiseLLLogit(final IntSeq target, final DataSet<? extends GroupedDSItem> owner) {
    this(target.stream().mapToDouble(i -> i), owner);
  }

  private PairwiseLLLogit(final DoubleStream target, final DataSet<? extends GroupedDSItem> owner) {
    final TObjectIntMap<String> groups = new TObjectIntHashMap<>();
    final List<IntSeqBuilder> builders = new ArrayList<>();
    int groupCount = 0;
    for (int i = 0; i < owner.length(); i++) {
      final String group = owner.at(i).groupId();
      int index = groups.get(group);
      if (index == groups.getNoEntryValue()) {
        groups.put(group, index = ++groupCount);
        builders.add(new IntSeqBuilder());
      }
      builders.get(index - 1).append(i);
    }

    this.groups = new int[groupCount][];
    for (int i = 0; i < builders.size(); i++) {
      final IntSeqBuilder builder = builders.get(i);
      this.groups[i] = builder.build().toArray();
    }
    this.target = target.collect(VecBuilder::new, VecBuilder::append, VecBuilder::addAll).build();
    this.owner = owner;
  }

  @Override
  public int dim() {
    return target.dim();
  }

  @Override
  public double value(final Vec point) {
    double result = 0;
    int count = 0;
    for (final int[] group : groups) {
      for (int i = 0; i < group.length; i++) {
        final int a = group[i];
        for (int j = i + 1; j < group.length; j++) {
          final int b = group[j];
          final double y = target.get(a) > target.get(b) ? 1 : -1;
          final double w = abs(target.get(a) - target.get(b));
          if (w > MathTools.EPSILON) {
            count++;
            result += -w * log(1. + exp(-y * (point.get(i) - point.get(j))));
          }
        }
      }
    }
    return exp(result/count);
  }

  @Override
  public Vec gradientTo(final Vec point, Vec to) {
    for (final int[] group : groups) {
      for (int i = 0; i < group.length; i++) {
        final int a = group[i];
        for (int j = i + 1; j < group.length; j++) {
          final int b = group[j];
          final double y = target.get(a) > target.get(b) ? 1 : -1;
          final double w = abs(target.get(a) - target.get(b));
          final double oneMinusP = 1. / (1. + exp(y * (point.get(i) - point.get(j))));
          to.adjust(i, -w * y * oneMinusP);
          to.adjust(j, w * y * oneMinusP);
        }
      }
    }
    return to;
  }


  public int label(final int idx) {
    return (int)target.get(idx);
  }

  public Vec labels() {
    return target;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
