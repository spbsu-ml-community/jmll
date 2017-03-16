package com.spbsu.exp.cart;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;

/**
 * Created by n_buga on 16.03.17.
 */
public class CARTSatL2 extends SatL2 {
    public CARTSatL2(Vec target, DataSet<?> base) {
        super(target, base);
    }

    @Override
    public double score(MSEStats stats) {
        return stats.weight > 2 ? super.score(stats): Double.POSITIVE_INFINITY;
    }
}
