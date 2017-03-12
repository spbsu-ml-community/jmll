package com.spbsu.exp.cart;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;

/**
 * Created by n_buga on 13.03.17.
 */
public class SteinEasy extends L2 {

    public SteinEasy(Vec target, DataSet<?> owner) {
        super(target, owner);
    }

    @Override
    public double bestIncrement(final MSEStats stats) {
        return stats.weight > MathTools.EPSILON ? stats.sum / (stats.weight + 1) : 0;
    }
}
