package com.expleague.ml.loss;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.GroupedDSItem;
import gnu.trove.map.TObjectDoubleMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.function.BiFunction;
import org.jetbrains.annotations.NotNull;

public class GroupedL2 extends L2 {

  private final Vec groupOffsets;
  public GroupedL2(final Vec target, final DataSet<? extends GroupedDSItem> owner) {
    this(target, owner, groupMeansOffsets());
  }

  //
  public GroupedL2(final Vec target, final DataSet<? extends GroupedDSItem> owner,
      BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec> offsetFunction) {
    super(target, owner);
    groupOffsets = offsetFunction.apply(target, owner);
    VecTools.scale(groupOffsets, -1);
  }

  public static BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec> randomOffsets(
      final double lowerBound, final double upperBound, final int seed) {
    return new BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec>() {
      private final double scale = upperBound - lowerBound;
      private final double offset = lowerBound;

      private Vec randomOffsets;
      private Vec lastTarget;
      private DataSet<? extends GroupedDSItem> lastDataSet;
      @Override
      public Vec apply(Vec target, DataSet<? extends GroupedDSItem> dataSet) {
        if (lastDataSet == dataSet && target == lastTarget) {
          return randomOffsets;
        }

        lastTarget = target;
        lastDataSet = dataSet;
        randomOffsets = new ArrayVec(lastTarget.dim());
        TObjectDoubleMap<String> groupOffsets = new TObjectDoubleHashMap<>();
        Random random = new Random();
        int idx = 0;
        for (GroupedDSItem item : lastDataSet) {
          String groupId = item.groupId();
          if (!groupOffsets.containsKey(groupId)) {
            groupOffsets.put(groupId, scale * random.nextDouble() + offset);
          }
          randomOffsets.set(idx++, groupOffsets.get(groupId));
        }

        return VecTools.copy(randomOffsets);
      }
    };
  }

  public static BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec> groupMeansOffsets() {
    return new BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec>() {
      private Vec lastTarget = null;
      private DataSet<? extends GroupedDSItem> lastDataSet = null;
      private Vec lastOffsets = null;

      @Override
      public Vec apply(Vec target, DataSet<? extends GroupedDSItem> dataSet) {
        // just check whether we received the same values as last time
        if (target == lastTarget && dataSet == lastDataSet) {
          return lastOffsets;
        }

        lastTarget = target;
        lastDataSet = dataSet;
        lastOffsets = new ArrayVec(lastTarget.dim());

        Map<String, TIntSet> groups = new HashMap<>();
        TObjectDoubleMap<String> groupSums = new TObjectDoubleHashMap<>();
        int idx = 0;
        for (GroupedDSItem item : dataSet) {
          String groupId = item.groupId();
          if (!groups.containsKey(groupId)) {
            groups.put(groupId, new TIntHashSet());
          }
          groups.get(groupId).add(idx);

          double targetAtIdx = target.get(idx);
          groupSums.adjustOrPutValue(groupId, targetAtIdx, targetAtIdx);
          ++idx;
        }

        groups.forEach((groupId, groupIndexes) -> {
          double groupMean = groupSums.get(groupId) / groupIndexes.size();
          groupIndexes.forEach(i -> {
            lastOffsets.set(i, groupMean);
            return true;
          });
        });

        return VecTools.copy(lastOffsets);
      }
    };
  }

  public static BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec> constantOffsets(
      final Vec offsets) {
    return new BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec>() {
      private final Vec offsetsSafeCopy = VecTools.copy(offsets);
      @Override
      public Vec apply(Vec target, DataSet<? extends GroupedDSItem> dataSet) {
        if (offsetsSafeCopy.dim() != target.dim()) {
          throw new IllegalArgumentException(String.format(
              "[ConstantOffsetsFunction] Dimensions of the target vector and offsets vector must be equal!\n"
                  + "Target  dim: %d\n"
                  + "Offsets dim: %d", target.dim(), offsets.dim()));
        }
        // TODO: do we need to copy?
        return VecTools.copy(offsetsSafeCopy);
      }
    };
  }
  @NotNull
  @Override
  public Vec gradient(final Vec x) {
    Vec gradient = VecTools.copy(x);
    VecTools.scale(gradient, -1);
    VecTools.append(gradient, target);
    VecTools.append(gradient, groupOffsets);
    VecTools.scale(gradient, -2);
    return gradient;
  }

  @Override
  public double value(final Vec point) {
    final Vec temp = VecTools.copy(point);
    VecTools.scale(temp, -1);
    VecTools.append(temp, target);
    VecTools.append(groupOffsets);
    return Math.sqrt(VecTools.sum2(temp) / temp.dim());
  }

  public Vec groupOffsets() {
    return VecTools.copy(groupOffsets);
  }
}
