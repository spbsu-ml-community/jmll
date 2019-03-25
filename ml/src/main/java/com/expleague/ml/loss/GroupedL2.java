package com.expleague.ml.loss;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.GroupedDSItem;
import com.expleague.ml.methods.GradientBoosting;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.jetbrains.annotations.NotNull;

public class GroupedL2 extends L2 {
  // So OK, we will make GroupedL2 (which is working name btw),
  // which loss and gradient depends on the value of 'x' not the target

  private final Set<TIntSet> groups = new HashSet<>();

  public GroupedL2(final Vec target, final DataSet<? extends GroupedDSItem> owner) {
    super(target, owner);
    Map<String, TIntSet> groupIndexesMap = new HashMap<>();
    int idx = 0;
    for (GroupedDSItem item : owner) {
      String groupId = item.groupId();
      if (!groupIndexesMap.containsKey(groupId)) {
        groupIndexesMap.put(groupId, new TIntHashSet());
      }
      groupIndexesMap.get(groupId).add(idx++);
    }
    groups.addAll(groupIndexesMap.values());
  }

  @NotNull
  @Override
  public Vec gradient(final Vec x) {
    Vec gradient = VecTools.copy(x);
    offsetByGroupMeans(gradient);
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

  private static double groupMean(final Vec x, final TIntSet group) {
    if (group.isEmpty()) {
      throw new IllegalArgumentException("Group must not be empty!");
    }

    double[] sum = {0};
    // TODO: can produce 'index out of bound exception' or smth like this
    group.forEach(i -> {
          sum[0] += x.get(i);
          return true;
        }
    );

    return sum[0] / group.size();
  }

  @Override
  public double value(Vec point) {
    return super.value(point);
  }
}
