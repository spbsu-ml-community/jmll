package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.optimization.impl.Nesterov2;
import junit.framework.TestCase;

/**
 * User: qdeee
 * Date: 09.09.13
 */
public class TensorNetFuncTest extends TestCase {
    private static final double EPS = 1e-4;

    public void testGradient() throws Exception {
        Mx X = new VecBasedMx(2, new ArrayVec(3, 4,
                                              6, 8));
        double c1 = 3.5;
        double c2 = 2.5;
        TensorNetFunction func = new TensorNetFunction(X, c1, c2);
        double[] actualGrad = func.gradient().value(new ArrayVec(0, 0, 0, 0)).toArray();
        double[] expectedGrad = new double[] {-2*c1*(X.get(0,0) + X.get(0,1)), -2*c1*(X.get(1,0) + X.get(1,1)),
                                              -2*c2*(X.get(0,0) + X.get(1,0)), -2*c2*(X.get(0,1) + X.get(1,1))};
        for (int i = 0; i < func.xdim() * 2; i++) {
            assertEquals(String.valueOf(i), expectedGrad[i], actualGrad[i], 1e-15);
        }
    }

    public void testOptimize1() throws Exception {
        int dim = 4;
        Mx X = new VecBasedMx(dim, new ArrayVec(4, 3, 2, 1,
                                                8, 6, 4, 2,
                                                12, 9, 6, 3,
                                                16, 12, 8, 4));
        double c1 = 2.165;
        double c2 = 4.105;
        TensorNetFunction func = new TensorNetFunction(X, c1, c2);

        Vec z0 = new ArrayVec(1,1,1,1,1,1,1,1);

        System.out.println("global convex : " + func.getGlobalConvexParam());
        System.out.println("lip : " + func.getGradLipParam());

        ConvexOptimize optimize = new Nesterov2(z0);
        Vec zMin = optimize.optimize(func, EPS);

        System.out.println("grad norm: " + VecTools.norm(func.gradient().value(zMin)));

        Vec u = new ArrayVec(dim);
        Vec v = new ArrayVec(dim);

        for (int i = 0; i < dim; i++) {
            u.set(i, zMin.get(i));
            v.set(i, zMin.get(i + dim));
        }

        System.out.println("U : " + u.toString());
        System.out.println("V : " + v.toString());
        System.out.println();
        System.out.println(VecTools.outer(u, v).toString());
        System.out.println("actual norm: " + VecTools.norm(VecTools.outer(u, v)));
        System.out.println("expected norm: " + VecTools.norm(X));
        assertEquals(VecTools.norm(X), VecTools.norm(VecTools.outer(u, v)), EPS);
    }

    public void testFindingParameters() {
        int dim = 4;
        Mx X = new VecBasedMx(dim, new ArrayVec(4, 3, 2, 1,
                                                8, 6, 4, 2,
                                                12, 9, 6, 3,
                                                16, 12, 8, 4));

        Vec z0 = new ArrayVec(1,1,1,1,1,1,1,1);
        ConvexOptimize nesterov = new Nesterov2(z0);

        Vec u = new ArrayVec(dim);
        Vec v = new ArrayVec(dim);

        double minRMSE = 10050000;
        double rmse_c1 = 0;
        double rmse_c2 = 0;

        for (double c1 = 0.005; c1 < 10; c1+=0.005) {
            for (double c2 = 0.005; c2 < 10; c2+=0.005) {
                ConvexFunction func = new TensorNetFunction(X, c1, c2);
                Vec zMin = nesterov.optimize(func, EPS);

                for (int i = 0; i < dim; i++) {
                    u.set(i, zMin.get(i));
                    v.set(i, zMin.get(i + dim));
                }
                Mx actualMx = VecTools.outer(u, v);

                double rmse = VecTools.distance(actualMx, X);
                if (rmse < minRMSE) {
                    minRMSE = rmse;
                    rmse_c1 = c1;
                    rmse_c2 = c2;
                }
            }
        }
        System.out.println("minimal error: " + minRMSE + "\t" + rmse_c1 + "\t" + rmse_c2);
    }
}