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

public class NormalizedDcg extends Func.Stub implements TargetFunc {
    private static final Logger LOG = LoggerFactory.getLogger(NormalizedDcg.class);
    private static final double LOG_2 = Math.log(2.);
    private static final double EPSILON = 1e-9;
    private final DataSet<? extends GroupedDSItem> owner;
    private final Vec target;
    private final List<Group> groups;

    public NormalizedDcg(final Vec target, final DataSet<? extends GroupedDSItem> owner) {
        this.owner = owner;
        this.target = target;

        Map<String, TIntList> groupsMap = new HashMap<>();
        int idx = 0;
        for (GroupedDSItem item : owner) {
            String groupId = item.groupId();
            if (!groupsMap.containsKey(groupId)) {
                groupsMap.put(groupId, new TIntArrayList());
            }
            groupsMap.get(groupId).add(idx++);
        }

        groups = new ArrayList<>();
        idx = 0;
        for (Map.Entry<String, TIntList> groupEntry : groupsMap.entrySet()) {
            int[] groupIdsArray = groupEntry.getValue().toArray();
            if (!hasDifferentLabels(target, groupIdsArray)) {
                LOG.warn(String.format("In the group with id [ %s ] all labels are the same!",
                    groupEntry.getKey()));
                continue;
            }
            double groupDcg = groupDcg(target, target, groupIdsArray);
            groups.add(new Group(groupDcg, groupIdsArray));
        }

        if (groups.isEmpty()) {
            throw new IllegalStateException("Received target labels has no different values!");
        }
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

    private static double dcg(final int[] labelsOrder, final Vec targetLabels) {
        double dcg = 0;
        for (int i = 0; i < labelsOrder.length; ++i) {
            double targetLabel = targetLabels.get(labelsOrder[i]);
            dcg += (Math.pow(2., targetLabel) - 1) / Math.log(i + 2);
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

        return summaryNdcg / groups.size();
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
