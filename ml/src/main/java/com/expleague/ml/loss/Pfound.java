package com.expleague.ml.loss;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.GroupedDSItem;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Pfound extends Func.Stub implements TargetFunc {
  private static final double DEFAULT_BREAK_PROBABILITY = 0.15;
  private static final double EPSILON = 1e-9;
  private static final Logger LOG = LoggerFactory.getLogger(Pfound.class);

  private final DataSet<? extends GroupedDSItem> owner;
  private final Vec target;
  private final double pBreak;
  private final List<int[]> groups;

  public Pfound(final Vec target, final DataSet<? extends GroupedDSItem> owner, final double pBreak) {
    this.owner = owner;
    this.target = target;
    this.pBreak = pBreak;

    Map<String, TIntList> groupsMap = new HashMap<>();
    int idx = 0;
    for (GroupedDSItem item : owner) {
      String groupId = item.groupId();
      groupsMap.putIfAbsent(groupId, new TIntArrayList());
      groupsMap.get(groupId).add(idx++);
    }

    groups = new ArrayList<>();
    for (Map.Entry<String, TIntList> groupEntry : groupsMap.entrySet()) {
      int[] groupIdsArray = groupEntry.getValue().toArray();
      if (!hasDifferentLabels(target, groupIdsArray)) {
        LOG.warn(String.format("In the group with id [ %s ] all labels are the same!",
            groupEntry.getKey()));
        continue;
      }
      groups.add(groupEntry.getValue().toArray());
    }

    if (groups.isEmpty()) {
      throw new IllegalStateException("Received target labels has no different values!");
    }
  }

  public Pfound(final Vec target, final DataSet<? extends GroupedDSItem> owner) {
    this(target, owner, DEFAULT_BREAK_PROBABILITY);
  }

  private static boolean hasDifferentLabels(final Vec target, final int[] groupIds) {
    double label = target.get(groupIds[0]);
    for (int i : groupIds) {
      if (Math.abs(label - target.get(i)) > EPSILON) {
        return true;
      }
    }
    return false;
  }

  private static double pfound(final int[] order, final Vec labels, final double pBreak) {
    double score = 0;
    double pStay = 1;
    for (int ord : order) {
      double relevance = labels.get(ord);
      score += pStay * relevance;
      pStay *= (1 - pBreak) * (1 - relevance);
    }

    return score;
  }

  private static double groupScore(final Vec ranks, final Vec labels, final int[] group,
      final double pBreak) {
    int[] order = Arrays.copyOf(group, group.length);
    double[] ranksCopy = new double[group.length];
    for (int i = 0; i < group.length; ++i) {
      ranksCopy[i] = ranks.get(group[i]);
    }
    ArrayTools.parallelSort(ranksCopy, order);

    return pfound(order, labels, pBreak);
  }

  @Override
  public DataSet<? extends GroupedDSItem> owner() {
    return owner;
  }

  @Override
  public double value(Vec ranks) {
    double pfoundSum = 0;
    for (int[] group : groups) {
      pfoundSum += groupScore(ranks, target, group, pBreak);
    }
    return pfoundSum / groups.size();
  }

  @Override
  public int dim() {
    return target.dim();
  }
}
