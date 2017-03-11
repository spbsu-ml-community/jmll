package com.spbsu.exp.cart;

import com.spbsu.commons.math.Trans;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import com.spbsu.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

/**
 * Created by n_buga on 10.03.17.
 */
public class CARTTreeOptimization<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
    private final int depth;
    private final BFGrid grid;

    public CARTTreeOptimization(final BFGrid grid, final int depth) {
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
                leaf.visitAllSplits((bf, left, right) -> scores[bf.bfIndex] = loss.score(left) + loss.score(right));
                final int bestSplit = ArrayTools.min(scores);
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
}
