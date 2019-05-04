package com.expleague.ml.loss;


import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.GroupedDSItem;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class NormalizedDcg extends Func.Stub implements TargetFunc {
    private static final double LOG_2 = Math.log(2.);
    private final DataSet<? extends GroupedDSItem> owner;
    private final Vec target;
    private final Group[] groups;

    public NormalizedDcg(Vec target, DataSet<? extends GroupedDSItem> owner) {
        this.owner = owner;
        this.target = target;

        Map<String, TIntList> groupsMap = new HashMap<>();
        int idx = 0;
        for (GroupedDSItem item : owner) {
            String groupId = item.groupId();
            if (!groupId.contains(groupId)) {
                groupsMap.put(groupId, new TIntArrayList());
            }
            groupsMap.get(groupId).add(idx++);
        }

        groups = new Group[groupsMap.size()];
        idx = 0;
        for (TIntList groupIds : groupsMap.values()) {
            int[] groupIdsArray = groupIds.toArray();
            double groupDcg = groupDcg(target, target, groupIdsArray);
            groups[idx++] = new Group(groupDcg, groupIdsArray);
        }
    }

    private static double dcg(final int[] labelsOrder, final Vec targetLabels) {
        double dcg = 0;
        for (int i = 0; i < labelsOrder.length; ++i) {
            double targetLabel = targetLabels.get(labelsOrder[i]);
            dcg += (Math.pow(2., targetLabel) - 1) / Math.log(i + 1);
        }
        return dcg;
    }

    private static double groupDcg(final Vec ranks, final Vec targetLabels, final int[] group) {
        final double[] invertedGroupRanks = new double[group.length];
        for (int i = 0; i < group.length; ++i) {
            invertedGroupRanks[i] = -ranks.get(group[i]);
        }

        final int[] order = Arrays.copyOf(group, group.length);
        ArrayTools.parallelSort(invertedGroupRanks, order);
        return dcg(order, targetLabels);
    }

    private static double groupNormalizedDcg(Vec ranks, Vec targetLabels, Group group) {
        return groupDcg(ranks, targetLabels, group.groupIdx) / group.groupDcg;
    }

    @Override
    public DataSet<?> owner() {
        return owner;
    }

    /**
     * Normalized DCG for a vector is calculated as mean NDCGs among groups
     * For each group DCG computed by the following formula:
     * sum_{i=1}^{p}{ (2^rel - 1) / log(i + 1) }
     *
     * @param ranks vector of ranks for corresponding items. It is assumed that these values determine the order of items
     * @return Normalized DCG produced by given ranks
     */
    @Override
    public double value(Vec ranks) {
        double summaryNdcg = 0;
        for (Group group : groups) {
            summaryNdcg += groupNormalizedDcg(ranks, target, group);
        }

        return summaryNdcg / groups.length;
    }

    @Override
    public int dim() {
        return target.dim();
    }

    private static class Group {
        final double groupDcg;
        final int[] groupIdx;

        Group(double groupDcg, int[] groupIdx) {
            this.groupDcg = groupDcg;
            this.groupIdx = groupIdx;
        }
    }
}
