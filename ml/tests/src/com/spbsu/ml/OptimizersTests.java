package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.optimization.ConvexOptimize;
import com.spbsu.ml.optimization.QuadraticFunction;
import com.spbsu.ml.optimization.TensorNetFunction;
import com.spbsu.ml.optimization.impl.ALS;
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

public class OptimizersTests extends TestCase {
    private static final double EPS = 1e-6;
    private static final int N = 6;
    private static final int TESTS_COUNT = 15;

    private static Logger LOG = Logger.create(OptimizersTests.class);

    public void testAllMethodsRandom() {
        Vec x0 = new ArrayVec(N);

        List<ConvexOptimize> algs = new ArrayList<ConvexOptimize>();
        algs.add(new Nesterov1(x0));
        algs.add(new Nesterov2(x0));
//        algs.add(new CustomNesterov(x0));
//        algs.add(new AdaptiveNesterov(x0));
        algs.add(new GradientDescent(x0));

        for (int k = 0; k < TESTS_COUNT; k++) {
            QuadraticFunction func = createRandomConvexFunc(new FastRandom(k));
            for (ConvexOptimize method : algs) {
                assertTrue(method.getClass().toString(), VecTools.distance(func.getExactExtremum(), method.optimize(func, EPS)) < EPS);
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
//        QuadraticFunction func = new QuadraticFunction(mxA, w, 0);
//        ConvexOptimize customNesterov = new CustomNesterov(new ArrayVec(3));
//        assertTrue(VecTools.distance(func.getExactExtremum(), customNesterov.optimize(func, EPS)) < EPS);
//    }
//
//    public void testAdaptiveNesterovRandom() {
//        QuadraticFunction func = createRandomConvexFunc(new FastRandom());
//        ConvexOptimize adaptiveNesterov = new AdaptiveNesterov(new ArrayVec(func.dim()));
//        Vec expected = func.getExactExtremum();
//        Vec actual = adaptiveNesterov.optimize(func, EPS);
//
//        LOG.message("|X| = " + VecTools.norm(actual));
//        assertTrue(VecTools.distance(expected, actual) < EPS);
//    }
//
//    public void testCustomNesterovRandom() {
//        QuadraticFunction func = createRandomConvexFunc(new FastRandom());
//        ConvexOptimize customNesterov = new CustomNesterov(new ArrayVec(func.dim()));
//        Vec expected = func.getExactExtremum();
//        Vec actual = customNesterov.optimize(func, EPS);
//
//        LOG.message("|X| = " + VecTools.norm(actual));
//        assertTrue(VecTools.distance(expected, actual) < EPS);
//    }

    public void testNesterov1Simple() {
        Mx mxA = new VecBasedMx(3, new ArrayVec(
                5, 0, 0,
                0, 15, 0,
                0, 0, 30));
        Vec w = new ArrayVec(-1, -1, -4);

        QuadraticFunction func = new QuadraticFunction(mxA, w, 0);
        ConvexOptimize nesterov1 = new Nesterov1(new ArrayVec(3));
        assertTrue(VecTools.distance(func.getExactExtremum(), nesterov1.optimize(func, EPS)) < EPS);
    }

    public void testNesterov2Simple() {
        Mx mxA = new VecBasedMx(3, new ArrayVec(
                5, 0, 0,
                0, 15, 0,
                0, 0, 30));
        Vec w = new ArrayVec(-1, -1, -4);

        QuadraticFunction func = new QuadraticFunction(mxA, w, 0);
        ConvexOptimize nesterov2 = new Nesterov2(new ArrayVec(3));
        assertTrue(VecTools.distance(func.getExactExtremum(), nesterov2.optimize(func, EPS)) < EPS);
    }

    public void testNesterov2Random() {
        QuadraticFunction func = createRandomConvexFunc(new FastRandom());
        ConvexOptimize nesterov2 = new Nesterov2(new ArrayVec(N));
        Vec expected = func.getExactExtremum();
        Vec actual = nesterov2.optimize(func, EPS);

        LOG.message("|X| = " + VecTools.norm(actual));
        assertTrue(VecTools.distance(expected, actual) < EPS);
    }

    public void testSolvingSystem() {
        Mx mxA = new VecBasedMx(2, new ArrayVec(
                2, 1,
                1, 2));
        QuadraticFunction func = new QuadraticFunction(mxA, new ArrayVec(-1, -11), 0);
        assertTrue(distance(func.getExactExtremum(), new ArrayVec(-3, 7)) < 1e-5);
    }

    private QuadraticFunction createRandomConvexFunc(Random rnd) {
        Vec w = new ArrayVec(N);
        Mx mxL = new VecBasedMx(N, N);
        Mx mxQ = new VecBasedMx(N, N);
        Mx mxC = new VecBasedMx(N, N);
        Mx sigma = new VecBasedMx(N, N);
        Mx mxA;

        for (int i = 0; i < mxC.dim(); i++)
            mxC.set(i, rnd.nextGaussian());                  //create random mx C

        VecTools.householderLQ(mxC, mxL, mxQ);

        for (int i = 0; i < mxL.rows(); i++)
            if (mxL.get(i, i) < 1e-3)
                mxL.set(i, i, 3 + rnd.nextDouble());         //make det(C) != 0

        mxC = VecTools.multiply(mxL, mxQ);

        for (int i = 0; i < sigma.rows(); i++) {
            sigma.set(i, i, rnd.nextDouble() + 10);          //make mxA positive-definite
        }

        mxA = VecTools.multiply(VecTools.multiply(mxC, sigma), VecTools.transpose(mxC));
        for (int i = 0; i < w.dim(); i++) {
            w.set(i, rnd.nextGaussian());
        }
        return new QuadraticFunction(mxA, w, 0);
    }

    public void testALS() throws Exception {
        int dim = 4;

        Mx X = new VecBasedMx(dim, new ArrayVec(4, 3, 2, 1,
                8, 6, 4, 2,
                12, 9, 6, 3,
                16, 12, 8, 4));
        double c1 = 6;
        double c2 = 6;
        TensorNetFunction func = new TensorNetFunction(X, c1, c2);

        Vec z0 = new ArrayVec(1, 1, 1, 1, 1, 1, 1, 1);
        ALS als = new ALS(z0, 1);
        Vec zMin = als.optimize(func);

        Vec u = new ArrayVec(dim);
        Vec v = new ArrayVec(dim);

        for (int i = 0; i < dim; i++) {
            u.set(i, zMin.get(i));
            v.set(i, zMin.get(i + dim));
        }

        System.out.println("u: " + u.toString());
        System.out.println("v: " + v.toString());
        assertTrue(VecTools.distance(X, VecTools.outer(u, v)) < 1e-5);
    }
}