package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.factorization.impl.ALS;
import com.spbsu.ml.optimization.FuncConvex;
import com.spbsu.ml.optimization.Optimize;
import com.spbsu.ml.optimization.PDQuadraticFunction;
import com.spbsu.ml.optimization.impl.GradientDescent;
import com.spbsu.ml.optimization.impl.Nesterov1;
import com.spbsu.ml.optimization.impl.Nesterov2;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.distance;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:07
 */

public class OptimizersTest extends TestCase {
    private static final double EPS = 1e-6;
    private static final int N = 6;
    private static final int TESTS_COUNT = 15;

    private static final Logger LOG = Logger.create(OptimizersTest.class);

    public void testAllMethodsRandom() {
        final Vec x0 = new ArrayVec(N);

        final List<Optimize<FuncConvex>> algs = new ArrayList<Optimize<FuncConvex>>();
        algs.add(new Nesterov1(x0, EPS));
        algs.add(new Nesterov2(x0, EPS));
//        algs.add(new CustomNesterov(x0));
//        algs.add(new AdaptiveNesterov(x0));
        algs.add(new GradientDescent(x0, EPS));

        for (int k = 0; k < TESTS_COUNT; k++) {
            final PDQuadraticFunction func = createRandomConvexFunc(new FastRandom(k));
            for (final Optimize<FuncConvex> method : algs) {
                assertTrue(method.getClass().toString(), VecTools.distance(func.getExactExtremum(), method.optimize(func)) < EPS);
            }
        }
    }

//    public void testCustomNesterovSimple() {
//        Mx mxA = new VecBasedMx(3, new ArrayVec(
//                5, 0, 0,
//                0, 15, 0,
//                0, 0, 30));
//        Vec w = new ArrayVec(-1, -1, -4);
//
//        PDQuadraticFunction func = new PDQuadraticFunction(mxA, w, 0);
//        ConvexOptimize customNesterov = new CustomNesterov(new ArrayVec(3));
//        assertTrue(VecTools.distance(func.getExactExtremum(), customNesterov.optimize(func, EPS)) < EPS);
//    }
//
//    public void testAdaptiveNesterovRandom() {
//        PDQuadraticFunction func = createRandomConvexFunc(new FastRandom());
//        ConvexOptimize adaptiveNesterov = new AdaptiveNesterov(new ArrayVec(func.ydim()));
//        Vec expected = func.getExactExtremum();
//        Vec actual = adaptiveNesterov.optimize(func, EPS);
//
//        LOG.message("|X| = " + VecTools.norm(actual));
//        assertTrue(VecTools.distance(expected, actual) < EPS);
//    }
//
//    public void testCustomNesterovRandom() {
//        PDQuadraticFunction func = createRandomConvexFunc(new FastRandom());
//        ConvexOptimize customNesterov = new CustomNesterov(new ArrayVec(func.ydim()));
//        Vec expected = func.getExactExtremum();
//        Vec actual = customNesterov.optimize(func, EPS);
//
//        LOG.message("|X| = " + VecTools.norm(actual));
//        assertTrue(VecTools.distance(expected, actual) < EPS);
//    }

    public void testNesterov1Simple() {
        final Mx mxA = new VecBasedMx(3, new ArrayVec(
                5, 0, 0,
                0, 15, 0,
                0, 0, 30));
        final Vec w = new ArrayVec(-1, -1, -4);

        final PDQuadraticFunction func = new PDQuadraticFunction(mxA, w, 0);
      final Optimize<FuncConvex> nesterov1 = new Nesterov1(new ArrayVec(3), EPS);
        assertTrue(VecTools.distance(func.getExactExtremum(), nesterov1.optimize(func)) < EPS);
    }

    public void testNesterov2Simple() {
        final Mx mxA = new VecBasedMx(3, new ArrayVec(
                5, 0, 0,
                0, 15, 0,
                0, 0, 30));
        final Vec w = new ArrayVec(-1, -1, -4);

        final PDQuadraticFunction func = new PDQuadraticFunction(mxA, w, 0);
        final Optimize<FuncConvex> nesterov2 = new Nesterov2(new ArrayVec(3), EPS);
        assertTrue(VecTools.distance(func.getExactExtremum(), nesterov2.optimize(func)) < EPS);
    }

    public void testNesterov2Random() {
        final PDQuadraticFunction func = createRandomConvexFunc(new FastRandom(N));
        final Optimize<FuncConvex> nesterov2 = new Nesterov2(new ArrayVec(N), EPS);
        final Vec expected = func.getExactExtremum();
        final Vec actual = nesterov2.optimize(func);

        LOG.message("|X| = " + VecTools.norm(actual));
        assertTrue(VecTools.distance(expected, actual) < EPS);
    }

    public void testSolvingSystem() {
        final Mx mxA = new VecBasedMx(2, new ArrayVec(
                2, 1,
                1, 2));
        final PDQuadraticFunction func = new PDQuadraticFunction(mxA, new ArrayVec(-1, -11), 0);
        assertTrue(distance(func.getExactExtremum(), new ArrayVec(-3, 7)) < 1e-5);
    }

    private PDQuadraticFunction createRandomConvexFunc(final Random rnd) {
        final Vec w = new ArrayVec(N);
        final Mx mxL = new VecBasedMx(N, N);
        final Mx mxQ = new VecBasedMx(N, N);
        Mx mxC = new VecBasedMx(N, N);
        final Mx sigma = new VecBasedMx(N, N);
        final Mx mxA;

        for (int i = 0; i < mxC.dim(); i++)
            mxC.set(i, rnd.nextGaussian());                  //create random mx C

        MxTools.householderLQ(mxC, mxL, mxQ);

        for (int i = 0; i < mxL.rows(); i++)
            if (mxL.get(i, i) < 1e-3)
                mxL.set(i, i, 3 + rnd.nextDouble());         //make det(C) != 0

        mxC = MxTools.multiply(mxL, mxQ);

        for (int i = 0; i < sigma.rows(); i++) {
            sigma.set(i, i, rnd.nextDouble() + 10);          //make mxA positive-definite
        }

        mxA = MxTools.multiply(MxTools.multiply(mxC, sigma), MxTools.transpose(mxC));
        for (int i = 0; i < w.dim(); i++) {
            w.set(i, rnd.nextGaussian());
        }
        return new PDQuadraticFunction(mxA, w, 0);
    }

    public void testALS() throws Exception {
        final int m = 3;
        final int n = 4;

        final Mx X = new VecBasedMx(n, new ArrayVec(
                8, 6, 4, 2,
                12, 9, 6, 3,
                16, 12, 8, 4));

        final ALS als = new ALS(1);
        final Pair<Vec, Vec> zMin = als.factorize(X);

        final Vec u = zMin.getFirst();
        final Vec v = zMin.getSecond();

        System.out.println("u: " + u.toString());
        System.out.println("v: " + v.toString());
        assertTrue(VecTools.distance(X, VecTools.outer(u, v)) < 1e-5);
    }
}