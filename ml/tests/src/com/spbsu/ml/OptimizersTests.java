package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.optimization.ConvexFunction;
import com.spbsu.ml.optimization.Optimize;
import com.spbsu.ml.optimization.QuadraticFunction;
import com.spbsu.ml.optimization.impl.GradientDescent;
import com.spbsu.ml.optimization.impl.Nesterov;
import junit.framework.TestCase;

import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:07
 */

public class OptimizersTests extends TestCase{
    private static final int N = 10;
    private static final int TESTS_COUNT = 100;
    private static Logger LOG = Logger.create(OptimizersTests.class);

    public void testOptimizeRandomMx() {
        Random rnd = new FastRandom();

        Optimize method1 = new Nesterov();
        Optimize method2 = new GradientDescent();

        Vec w =  new ArrayVec(N);
        Vec b = new ArrayVec(N);
        Mx mxL = new VecBasedMx(N, N);
        Mx mxQ = new VecBasedMx(N, N);
        Mx mxC = new VecBasedMx(N, N);
        Mx sigma = new VecBasedMx(N, N);
        Mx mxA;

        for (int k = 0; k < TESTS_COUNT; k++) {
            {
                for (int i = 0; i < mxC.dim(); i++)
                    mxC.set(i, rnd.nextGaussian());

                VecTools.householderLQ(mxC, mxL, mxQ);

                for (int i = 0; i < mxL.rows(); i++)
                    if (mxL.get(i, i) < 1e-3)
                        mxL.set(i, i, 3 + rnd.nextDouble());

                mxC = VecTools.multiply(mxL, mxQ);

                for (int i = 0; i < sigma.rows(); i++) {
                    sigma.set(i, i, rnd.nextDouble() + 10);
                }

                mxA = VecTools.multiply(VecTools.multiply(mxC, sigma), VecTools.transpose(mxC));
                for (int i = 0; i < w.dim(); i++) {
                    w.set(i, rnd.nextGaussian());
                    b.set(i, -1 * w.get(i));
                }
            }

            ConvexFunction func = new QuadraticFunction(mxA, w, 0);

            Vec x = solveSystem(mxA, b);
            LOG.message("|x| = " + VecTools.norm(x));

            Vec min1 = method1.optimize(func, 1e-4);
            Vec min2 = method2.optimize(func, 1e-4);

            assertTrue("Nesterov", distance(x, min1) < 1e-4);
            assertTrue("GradDesc", distance(x, min2) < 1e-4);
        }
    }


    public void testOptimize1() {

        Mx mxA = new VecBasedMx(3, new ArrayVec(
                1, 0, 0,
                1, 1, 0,
                1, 2, 3));
        Vec w = new ArrayVec(-1, -1, -4);
        double w0 = 1.5;

        QuadraticFunction func = new QuadraticFunction(mxA, w, w0);
        Optimize method1 = new Nesterov();
        Optimize method2 = new GradientDescent();
        Vec min1 = method1.optimize(func, 0.01);
        Vec min2 = method2.optimize(func, 0.01);

        double distance1 = distance(min1, new ArrayVec(1, 0, 1));
        assertTrue(distance1 < 1e-2);

        double distance2 = distance(min2, new ArrayVec(1, 0, 1));
        assertTrue(distance2 < 1e-2);
    }


    public void testSolvingSystem() {
        Mx mxA = new VecBasedMx(2, new ArrayVec(
                2, 1,
                1, 2));
        Vec b = new ArrayVec(1, 11);
        Vec x = solveSystem(mxA, b);

        assertTrue(distance(x, new ArrayVec(-3, 7)) < 1e-5);
    }

    private Vec solveSystem(Mx mxA, Vec b) {
        Mx l = VecTools.choleskyDecomposition(mxA);
        Mx inverse = VecTools.inverseLTriangle(l);
        Vec x = VecTools.multiply(VecTools.multiply(VecTools.transpose(inverse), inverse), b);

        //stupid cast (VecBasedMx -> ArrayVec) for beautiful solution out
        Vec result = new ArrayVec(b.dim());
        for (int i = 0; i < x.dim(); i++)
            result.set(i, x.get(i));
        return result;
    }
}
