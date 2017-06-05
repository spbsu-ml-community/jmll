package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.greedyRegion.BFOptimizationRegion;
import gnu.trove.list.array.TIntArrayList;

/**
 * Created by au-rikka on 29.04.17.
 */
public class BFOptimizationSimpleRegion extends BFOptimizationRegion {
//    protected double[] weights;

    public BFOptimizationSimpleRegion(BinarizedDataSet bds, StatBasedLoss oracle, int[] points) {
        super(bds, oracle, points);
        this.aggregate = new Aggregate(bds, oracle.statsFactory(), points);
//        this.weights = weights;
    }

    public int[] getPoints() {
        return pointsInside;
    }

//    @Override
//    public void split(final BFGrid.BinaryFeature feature, final boolean mask) {
//        final byte[] bins = bds.bins(feature.findex);
//
//        final TIntArrayList newInside = new TIntArrayList();
//        final TIntArrayList newOutside = new TIntArrayList();
//
//
//        for (int index : pointsInside) {
//            if ((bins[index] > feature.binNo) != mask) {
//                newOutside.add(index);
//            } else {
//                newInside.add(index);
//            }
//        }
//        pointsInside = newInside.toArray();
//        if (newInside.size() < newOutside.size()) {
//            aggregate = new Aggregate(bds, oracle.statsFactory(), pointsInside, weights);
//        } else {
//            aggregate.remove(new BFOptimizationSimpleRegion(bds, oracle, newOutside.toArray(), weights).aggregate);
//        }
//    }

}
