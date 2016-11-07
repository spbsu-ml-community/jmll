package com.spbsu.ml.methods.cart;

import com.spbsu.commons.math.vectors.Vec;

/**
 * Created by n_buga on 17.10.16.
 */
public class Condition {
//    final private double eps = 10e-6;

    private int featureNo;
    private double barrier;
    private boolean less;

    public Condition() { }

    public Condition(int featureNo, double barrier, boolean less) {
        this.featureNo = featureNo;
        this.barrier = barrier;
        this.less = less;
    }

    public Condition(Condition c) {
        this.featureNo = c.featureNo;
        this.barrier = c.barrier;
        this.less = c.less;
    }

    public boolean checkFeature(Vec x) {
        double a = x.at(featureNo);
        if (less)
            return a < barrier;
        else
            return a >= barrier;
    }

    public Condition set(int featureNo, double barrier, boolean less) {
        this.featureNo = featureNo;
        this.barrier = barrier;
        this.less = less;
        return this;
    }

    public Condition set(boolean less) {
        this.less = less;
        return this;
    }

    public int getFeatureNo() {
        return featureNo;
    }
}
