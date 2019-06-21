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
//        continue;
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

  /**
   * Calculates Pfound metric for a single query. See:
   *
   * Gulin A.; Karpovich P.; Raskovalov D.; Segalovich I. (2009),
   * "Yandex at ROMIP'2009: optimization of ranking algorithms by machine learning methods",
   * Proceedings of ROMIP'2009: 163–168 (in Russian)
   *
   * for more details
   * @param order order of pages determined by predicted ranks
   * @param labels probabilities of pages to be relevant ot a query
   * @param pBreak probability of an user to suddenly stop looking for page
   * @return pfound score
   */
  private static double pfound(final int[] order, final Vec labels, final double pBreak) {
    double score = 0;
    double pLook = 1;
    for (int ord : order) {
      double pRel = labels.get(ord);
      score += pLook * pRel;
      pLook *= (1 - pBreak) * (1 - pRel);
    }

    return score;
  }

  private static double groupScore(final Vec ranks, final Vec labels, final int[] group,
      final double pBreak) {
    int[] order = Arrays.copyOf(group, group.length);
    double[] ranksCopy = new double[group.length];
    for (int i = 0; i < group.length; ++i) {
      ranksCopy[i] = -ranks.get(group[i]);
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
