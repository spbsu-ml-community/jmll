package com.expleague.ml.data.ctrs;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.ml.GridTools;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;

public class CtrTarget {
  private Vec target;
  private CtrTargetType type;

  public CtrTarget(final Vec target,
                   final CtrTargetType type) {
    if (type == CtrTargetType.Binomial) {
      double[] values = target.toArray();
      Arrays.sort(values);
      final TIntArrayList borders = GridTools.greedyLogSumBorders(values, 1);
      final double border = values[borders.get(0) - 1];
      VecBuilder builder = new VecBuilder(target.dim());
      for (int i = 0; i < target.dim(); ++i) {
        builder.append(target.get(i) > border ? 1.0 : 0.0);
      }
      this.target = builder.build();
    } else {
      throw new RuntimeException("unimplemented");
    }
    this.type = type;
  }

  public Vec target() {
    return target;
  }

  public CtrTargetType type() {
    return type;
  }

  public enum CtrTargetType {
    Binomial,
    Normal
  }
}