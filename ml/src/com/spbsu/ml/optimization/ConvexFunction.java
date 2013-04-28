package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Oracle1;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:01
 */

public abstract class ConvexFunction implements Oracle1 {

    private final int dim;

    private final double m;   //convex parameter
    private final double lk;   //Lipshitz constant

    public abstract double value(Vec x);
    public abstract Vec gradient(Vec x);

    protected ConvexFunction(int dim, double m, double lk) {
        this.dim = dim;
        this.m = m;
        this.lk = lk;
    }

    protected ConvexFunction(int dim, double[] funcParams) {
        this(dim, funcParams[0], funcParams[1]);
    }

    public int getDim() {
        return dim;
    }

    public double getConvexParam() {
        return m;
    }

    public double getLk() {
        return lk;
    }
}
