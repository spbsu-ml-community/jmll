package com.expleague.ml.loss;


import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.meta.GroupedDSItem;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

public class NormalizedDcg extends Func.Stub implements TargetFunc {
    private static final double LOG_2 = Math.log(2.);
    private final DataSet<? extends GroupedDSItem> owner;

    private final RankedLabel[] rankedLabels;
    private final Group[] groups;

    public NormalizedDcg(Vec target, DataSet<? extends GroupedDSItem> owner) {
        this.owner = owner;

        rankedLabels = new RankedLabel[target.dim()];
        for (int i = 0; i < rankedLabels.length; ++i) {
            rankedLabels[i] = new RankedLabel(target.get(i));
        }

        Map<String, TIntSet> groupsMap = new HashMap<>();
        int idx = 0;
        for (GroupedDSItem item : owner) {
            String groupId = item.groupId();
            if (!groupId.contains(groupId)) {
                groupsMap.put(groupId, new TIntHashSet());
            }
            groupsMap.get(groupId).add(idx++);
        }

        groups = new Group[groupsMap.size()];
        idx = 0;
        for (TIntSet groupIds : groupsMap.values()) {
            int[] groupIdsArray = groupIds.toArray();
            double groupDcg = groupDcg(rankedLabels, groupIdsArray);
            groups[idx++] = new Group(groupDcg, groupIdsArray);
        }
    }

    private static double dcg(final RankedLabel[] rankedLabels) {
        double dcg = 0;
        for (int i = 0; i < rankedLabels.length; ++i) {
            dcg += (Math.pow(2., rankedLabels[i].label()) - 1) / Math.log(i + 1);
        }
        return dcg;
    }

    private static double groupDcg(final RankedLabel[] rankedLabels, int[] group) {
        RankedLabel[] groupRankedLabels = new RankedLabel[group.length];
        for (int i = 0; i < group.length; ++i) {
            groupRankedLabels[i] = rankedLabels[group[i]];
        }
        Arrays.sort(groupRankedLabels, Comparator.<RankedLabel>comparingDouble(RankedLabel::rank).reversed());
        return dcg(groupRankedLabels);
    }

    private static double groupNormalizedDcg(final RankedLabel[] rankedLabels, Group group) {
        int[] groupIds = group.groupIdx;
        RankedLabel[] groupRankedLabels = new RankedLabel[groupIds.length];
        for (int i = 0; i < groupIds.length; ++i) {
            groupRankedLabels[i] = rankedLabels[groupIds[i]];
        }
        Arrays.sort(groupRankedLabels, Comparator.<RankedLabel>comparingDouble(RankedLabel::rank).reversed());
        return dcg(groupRankedLabels) / group.groupDcg;
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
        int dim = dim();
        for (int i = 0; i < dim; ++i) {
            RankedLabel rankedLabel = rankedLabels[i];
            rankedLabel.rank(ranks.get(i));
        }

        double summaryNdcg = 0;
        for (Group group : groups) {
            summaryNdcg += groupNormalizedDcg(rankedLabels, group);
        }

        return summaryNdcg / groups.length;
    }

    @Override
    public int dim() {
        return rankedLabels.length;
    }

    private static class Group {
        final double groupDcg;
        final int[] groupIdx;

        Group(double groupDcg, int[] groupIdx) {
            this.groupDcg = groupDcg;
            this.groupIdx = groupIdx;
        }
    }

    private static class RankedLabel {
        private final double label;
        private double rank;

        RankedLabel(double label) {
            this(label, label);
        }

        RankedLabel(double rank, double label) {
            this.rank = rank;
            this.label = label;
        }

        double rank() {
            return rank;
        }

        double label() {
            return label;
        }

        void rank(double rank) {
            this.rank = rank;
        }
    }
}
