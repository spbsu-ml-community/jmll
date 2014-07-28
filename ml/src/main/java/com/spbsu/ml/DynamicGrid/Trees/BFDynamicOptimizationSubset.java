package com.spbsu.ml.DynamicGrid.Trees;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.DynamicGrid.AggregateDynamic;
import com.spbsu.ml.DynamicGrid.Interface.BinaryFeature;
import com.spbsu.ml.data.impl.BinarizedDynamicDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import gnu.trove.list.array.TIntArrayList;


@SuppressWarnings("unchecked")
public class BFDynamicOptimizationSubset {
    private final BinarizedDynamicDataSet bds;
    public int[] points;
    private final StatBasedLoss<AdditiveStatistics> oracle;
    private AggregateDynamic aggregate;

    public BFDynamicOptimizationSubset(BinarizedDynamicDataSet bds, StatBasedLoss oracle, int[] points) {
        this.bds = bds;
        this.points = points;
        this.oracle = oracle;
        this.aggregate = new AggregateDynamic(bds, oracle.statsFactory(), points);
    }

    public BFDynamicOptimizationSubset split(BinaryFeature feature) {
        TIntArrayList left = new TIntArrayList(points.length);
        TIntArrayList right = new TIntArrayList(points.length);
        final int[] bins = bds.bins(feature.fIndex());
        for (int i : points) {
            if (bins[i] <= feature.binNo()) {
                left.add(i);
            } else {
                right.add(i);
            }
        }
        final BFDynamicOptimizationSubset rightBro = new BFDynamicOptimizationSubset(bds, oracle, right.toArray());
        aggregate.remove(rightBro.aggregate);
        points = left.toArray();
        aggregate.updatePoints(points);
//        AggregateDynamic test = new AggregateDynamic(bds, oracle.statsFactory(), points);
//        compareAgregate(aggregate,test);
        return rightBro;
    }

    public static boolean compareAgregate(AggregateDynamic firstAgr, AggregateDynamic secondAgr) {
        boolean equals = true;
        for (int f = 0; f < firstAgr.bins.length && equals; ++f)
            for (int bin = 0; bin < firstAgr.bins[f].length; ++bin) {
                L2.MSEStats first = (L2.MSEStats) (((WeightedLoss.Stat) firstAgr.bins[f][bin])).inside;
                L2.MSEStats second = (L2.MSEStats) (((WeightedLoss.Stat) secondAgr.bins[f][bin])).inside;
                if (Math.abs(first.sum - second.sum) > 1e-9 || Math.abs(first.sum2 - second.sum2) > 1e-9) {
                    equals = false;
                    break;
                }
            }
        if (!equals) {
            System.out.println("aaa");
        }
        return equals;
    }

    public int size() {
        return points.length;
    }

    public void visitAllSplits(AggregateDynamic.SplitVisitor<? extends AdditiveStatistics> visitor) {
        aggregate.visit(visitor);
    }

    public <T extends AdditiveStatistics> void visitSplit(BinaryFeature bf, AggregateDynamic.SplitVisitor<T> visitor) {
        final T left = (T) aggregate.combinatorForFeature(bf);
        final T right = (T) oracle.statsFactory().create().append(aggregate.total()).remove(left);
        visitor.accept(bf, left, right);
    }

    public AdditiveStatistics total() {
        return aggregate.total();
    }


    public void rebuild(int... features) {
//        this.aggregate = new AggregateDynamic(bds, oracle.statsFactory(), points);
        this.aggregate.rebuild(features);
//        AggregateDynamic test = new AggregateDynamic(bds, oracle.statsFactory(), points);
//
//        boolean equals = true;
//
//
//
//
//        for (int f = 0; f < aggregate.bins.length && equals; ++f)
//            for (int bin = 0; bin < aggregate.bins[f].length; ++bin) {
//                L2.MSEStats first = (L2.MSEStats) (((WeightedLoss.Stat) aggregate.bins[f][bin])).inside;
//                L2.MSEStats second = (L2.MSEStats) (((WeightedLoss.Stat) test.bins[f][bin])).inside;
//                if (Math.abs(first.sum - second.sum) > 1e-9 || Math.abs(first.sum2 - second.sum2) > 1e-9) {
//                    equals = false;
//                    break;
//                }
//            }
//        if (!equals) {
//            System.out.println("aaa");
//        }

    }
}
