package com.spbsu.exp.cart;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by n_buga on 10.03.17.
 */
public class CARTTreeOptimization<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
    private final int depth;
    private final BFGrid grid;
    private final double lambda;

    public CARTTreeOptimization(final BFGrid grid, final int depth) {
        this(grid, depth, 0);
    }

    public CARTTreeOptimization(final BFGrid grid, final int depth, final double lambda) {
        this.lambda = lambda;
        this.grid = grid;
        this.depth = depth;
    }

    @Override
    public CARTTree fit(VecDataSet learn, Loss loss) {
        Pair<List<BFOptimizationSubset>, List<LeafConditions>> result = findBestSubsets(learn,loss);
        List<BFOptimizationSubset> leaves = result.getFirst();
        List<LeafConditions> conditions = result.getSecond();
        final double[] step = new double[leaves.size()];
        final double[] based = new double[leaves.size()];
        for (int i = 0; i < step.length; i++) {
            step[i] = loss.bestIncrement(leaves.get(i).total());
            based[i] = leaves.get(i).size();
        }
        return new CARTTree(conditions, step, based);
    }

    public final Pair<List<BFOptimizationSubset>, List<LeafConditions>> findBestSubsets(final VecDataSet ds, final Loss loss) {
        List<BFOptimizationSubset> leaves = new ArrayList<>(1 << depth);
        List<LeafConditions> leafConditions = new ArrayList<>(1 << depth);
        final BinarizedDataSet bds =  ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

        leaves.add(new BFOptimizationSubset(bds, loss, learnPoints(loss, ds)));
        leafConditions.add(new LeafConditions());

        final double[] scores = new double[grid.size()];

        for (int level = 0; level < depth; level++) {
            final List<BFOptimizationSubset> next = new ArrayList<>(leaves.size() * 2);
            final List<LeafConditions> nextCond = new ArrayList<>(leaves.size() * 2);
            for (int leafNum = 0; leafNum < leaves.size(); leafNum++) {
                final BFOptimizationSubset leaf = leaves.get(leafNum);
                final int leavesSize = leaves.size();
                final int dsSize = ds.length();
                leaf.visitAllSplits((bf, left, right) ->
                        scores[bf.bfIndex] =
                                (loss.score(left) + loss.score(right))*
                                        (1 + lambda*reg(left, right, leavesSize, dsSize)));
                final int bestSplit = ArrayTools.min(scores);
                final double score = loss.score(leaf.total());
                if (bestSplit < 0 || scores[bestSplit] >= loss.score(leaf.total())) {
                    next.add(leaf);
                    nextCond.add(leafConditions.get(leafNum));
                    continue;
                }

                final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);

                next.add(leaf);
                nextCond.add(new LeafConditions(leafConditions.get(leafNum), bestSplitBF, false));
                next.add(leaf.split(bestSplitBF));
                nextCond.add(new LeafConditions(leafConditions.get(leafNum), bestSplitBF, true));
            }
            leaves = next;
            leafConditions = nextCond;
        }
        return new Pair<>(leaves, leafConditions);
    }

    private int[] learnPoints(Loss loss, VecDataSet ds) {
        if (loss instanceof WeightedLoss) {
            return ((WeightedLoss) loss).points();
        } else return ArrayTools.sequence(0, ds.length());
    }

    private double reg(AdditiveStatistics left, AdditiveStatistics right, int leavesSize, int dsSize) {
        L2.MSEStats leftMSE = (L2.MSEStats) ((WeightedLoss.Stat) left).inside;
        L2.MSEStats rightMSE = (L2.MSEStats) ((WeightedLoss.Stat) right).inside;
        double genCount = leftMSE.weight + rightMSE.weight;
        double c = 1.0/(genCount + 2)*(genCount + 1)/(dsSize + leavesSize);
        double p1 = c*(leftMSE.weight + 1);
        double p2 = c*(rightMSE.weight + 1);
        return -(-p1*Math.log(p1) - p2*Math.log(p2));
    }
}
