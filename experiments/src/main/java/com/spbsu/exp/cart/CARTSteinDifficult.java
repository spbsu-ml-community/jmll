package com.spbsu.exp.cart;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;

/**
 * Created by n_buga on 13.03.17.
 */
public class CARTSteinDifficult extends CARTL2 {
    public CARTSteinDifficult(Vec target, DataSet<?> owner) {
        super(target, owner);
    }

    @Override
    public double bestIncrement(final MSEStats stats) {
        if (stats.weight <= 2 || stats.sum2 < 1e-6)
            return stats.sum/(stats.weight);
        return (1 - (stats.weight - 2)*score(stats)/stats.sum2)*(stats.sum/stats.weight);
    }

}
