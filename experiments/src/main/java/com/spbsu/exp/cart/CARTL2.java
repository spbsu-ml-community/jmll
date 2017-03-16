package com.spbsu.exp.cart;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;

/**
 * Created by n_buga on 16.03.17.
 */
public class CARTL2 extends L2 {

    public CARTL2(Vec target, DataSet<?> owner) {
        super(target, owner);
    }

    @Override
    public double score(MSEStats stats) {
        return stats.weight > 2 ? super.score(stats): Double.POSITIVE_INFINITY;
    }
}
