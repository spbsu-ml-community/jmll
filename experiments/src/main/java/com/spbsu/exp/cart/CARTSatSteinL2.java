package com.spbsu.exp.cart;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;

/**
 * Created by n_buga on 18.03.17.
 */
public class CARTSatSteinL2 extends SatL2 {
    public CARTSatSteinL2(Vec target, DataSet<?> owner) {
        super(target, owner);
    }

    @Override
    public double bestIncrement(final MSEStats stats) {
        if (stats.weight <= 2 || stats.sum2 < 1e-6)
            return super.bestIncrement(stats);
        return (1 - (stats.weight - 2)*score(stats)/(stats.sum2*stats.weight))*(stats.sum/stats.weight);
    }
}
