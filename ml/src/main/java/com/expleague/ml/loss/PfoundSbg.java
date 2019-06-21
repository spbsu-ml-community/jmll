package com.expleague.ml.loss;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.GroupedDSItem;
import com.expleague.ml.meta.items.QURLItem;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PfoundSbg extends Func.Stub implements TargetFunc {
  private static final double DEFAULT_BREAK_PROBABILITY = 0.15;
  private static final double EPSILON = 1e-9;
  private static final Logger LOG = LoggerFactory.getLogger(PfoundSbg.class);

  private final DataSet<? extends GroupedDSItem> owner;
  private final Vec target;
  private final double pBreak;
  private final List<Group> groups;


  public PfoundSbg(final Vec target, final DataSet<? extends QURLItem> owner, final double pBreak) {
    this.owner = owner;
    this.target = target;
    this.pBreak = pBreak;

    Map<String, Map<String, TIntList>> subgroupsMap = new HashMap<>();

    int idx = 0;
    for (QURLItem item : owner) {
      String groupId = item.groupId();
      String subgroupId = item.subgroupId();

      subgroupsMap.putIfAbsent(groupId, new HashMap<>());
      Map<String, TIntList> sbg = subgroupsMap.get(groupId);
      sbg.putIfAbsent(subgroupId, new TIntArrayList());
      sbg.get(subgroupId).add(idx++);
    }

    LOG.info(String.format("Groups found: %d\nOverall elements: %d", subgroupsMap.size(), idx));
    groups = new ArrayList<>();
    for (Map<String, TIntList> sbg : subgroupsMap.values()) {
      int[][] subgroups = new int[sbg.size()][];
      idx = 0;
      for (TIntList ids : sbg.values()) {
        subgroups[idx++] = ids.toArray();
      }
      groups.add(new Group(subgroups));
    }

    if (groups.isEmpty()) {
      throw new IllegalStateException("Received target labels has no different values!");
    }
  }

  public PfoundSbg(final Vec target, final DataSet<? extends QURLItem> owner) {
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
   * for the open version of the metric. In current version subgroups also are taken
   * into account
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

  private static int argMax(int[] args, Vec values) {
    int argMax = args[0];
    for (int i : args) {
      if (values.get(i) > values.get(argMax)) {
        argMax = i;
      }
    }
    return argMax;
  }

  private static double groupScore(final Vec ranks, final Vec labels, final Group group,
      final double pBreak) {
    int[] order = new int[group.subgroupsCount()];
    for (int i = 0; i < order.length; ++i) {
      order[i] = argMax(group.subgroupIds[i], ranks);
    }
    double[] ranksCopy = new double[order.length];
    for (int i = 0; i < order.length; ++i) {
      ranksCopy[i] = -ranks.get(order[i]);
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
    for (Group group : groups) {
      pfoundSum += groupScore(ranks, target, group, pBreak);
    }
    return pfoundSum / groups.size();
  }

  @Override
  public int dim() {
    return target.dim();
  }

  private static final class Group {
    final int[][] subgroupIds;

    public Group(int[][] subgroupIds) {
      this.subgroupIds = subgroupIds;
    }

    int subgroupsCount() {
      return subgroupIds.length;
    }
  }
}
