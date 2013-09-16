package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;


/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:03
 */

public class QuadraticFunction implements ConvexFunction{
    private final Mx mxA;
    private final Vec w;
    private final double w0;

    private final int dim;

    private final double m;
    private final double l;

    // result[0] = m (convex param),
    // result[1] = l (lipshitz const);
    private static double[] getConvAndLipConstants(Mx mxA, Vec w) {
        Mx q = new VecBasedMx(mxA.rows(), mxA.columns());
        Mx sigma = new VecBasedMx(mxA.rows(), mxA.columns());
        VecTools.eigenDecomposition(mxA, q, sigma);

        double minEigenValue = sigma.get(0, 0);
        double maxEigenValue = sigma.get(0, 0);
        for (int i = 1; i < sigma.rows(); i++) {
            if (sigma.get(i, i) < minEigenValue)
                minEigenValue = sigma.get(i, i);
            if (sigma.get(i, i) > maxEigenValue)
                maxEigenValue = sigma.get(i, i);
        }
        return new double[]{minEigenValue, maxEigenValue};
    }

    public QuadraticFunction(Mx mxA, Vec w, double w0) {
        double[] params = getConvAndLipConstants(mxA, w);

        this.mxA = mxA;
        this.w = w;
        this.w0 = w0;

        this.dim = w.dim();

        this.m = params[0];
        this.l = params[1];
    }

    @Override
    public Vec gradient(Vec x) {
        return VecTools.append(VecTools.multiply(mxA, x), w);
    }

    @Override
    public double value(Vec x) {
        return VecTools.multiply(VecTools.multiply(mxA, x), x) + VecTools.multiply(w, x) + w0;
    }

    public double getQuadrPartValue(Vec x) {
        return VecTools.multiply(VecTools.multiply(mxA, x), x);
    }

    public Vec getW() {
        return w;
    }

    public Vec getExactExtremum() {
        Vec b = VecTools.copy(w);
        VecTools.scale(b, -1.0);
//
//        Mx L = new VecBasedMx(dim, dim);
//        Mx Q = new VecBasedMx(dim, dim);
//        VecTools.householderLQ(mxA, L, Q);
//        return VecTools.multiply(VecTools.multiply(VecTools.transpose(Q), VecTools.inverseLTriangle(L)), b);
        Mx l = VecTools.choleskyDecomposition(mxA);
        Mx inverse = VecTools.inverseLTriangle(l);
        Vec x = VecTools.multiply(VecTools.multiply(VecTools.transpose(inverse), inverse), b);

        //stupid cast (VecBasedMx -> ArrayVec) for solution out
        Vec result = new ArrayVec(b.dim());
        for (int i = 0; i < x.dim(); i++)
            result.set(i, x.get(i));
        return result;
    }

    @Override
    public int dim() {
        return dim;
    }

    @Override
    public double getLocalConvexParam(Vec x) {
        return m;
    }

    @Override
    public double getGradLipParam() {
        return l;
    }

    @Override
    public double getGlobalConvexParam() {
        return m;
    }
}
