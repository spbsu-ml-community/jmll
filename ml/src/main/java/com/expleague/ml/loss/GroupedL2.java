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
import java.util.function.BiFunction;
import org.jetbrains.annotations.NotNull;

public class GroupedL2 extends L2 {

  private final Vec groupsOffsets;
  public GroupedL2(final Vec target, final DataSet<? extends GroupedDSItem> owner) {
    this(target, owner, groupMeansOffsets());
  }

  //
  public GroupedL2(final Vec target, final DataSet<? extends GroupedDSItem> owner,
      BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec> offsetFunction) {
    super(target, owner);
    groupsOffsets = offsetFunction.apply(target, owner);
  }

  public static BiFunction<Vec, DataSet<? extends GroupedDSItem>, Vec> randomOffsets(
      final double lowerBound, final double upperBound, final int seed) {
    return (vec, ds) -> {

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
    final Vec safeCopy = VecTools.copy(offsets);
    return (v, ds) -> VecTools.copy(safeCopy);
  }
  @NotNull
  @Override
  public Vec gradient(final Vec x) {
    Vec gradient = VecTools.copy(x);
    VecTools.scale(gradient, -1);
    VecTools.append(gradient, target);
    VecTools.scale(gradient, -2);
    return gradient;
  }

  /**
   * We assume that elements of 'x' form group, which were determined by the 'owner' dataset
   * Method subtracts from each element of 'x' its group mean.
   * @param x vector to be modified
   */
  private void offsetByGroupMeans(Vec x) {
    for (TIntSet group : groups) {
      double negativeGroupMean = -groupMean(x, group);
      group.forEach(i -> {
            x.adjust(i, negativeGroupMean);
            return true;
          }
      );
    }
  }

  @Override
  public double value(Vec point) {
    return super.value(point);
  }
}
