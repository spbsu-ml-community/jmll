package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;


/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:03
 */

public class QuadraticFunction extends ConvexFunction{
    private final Mx mxA;
    private final Vec w;
    private final double w0;

    public QuadraticFunction(Mx mxA, Vec w, double w0) {
        super(w.dim(), getConvAndLipConstants(mxA, w));

        this.mxA = mxA;
        this.w = w;
        this.w0 = w0;
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

    // result[0] = m (convex param),
    // result[1] = lk (lipshitz const);
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
        return new double[]{minEigenValue, 2 * maxEigenValue + VecTools.norm(w)};
    }
}
