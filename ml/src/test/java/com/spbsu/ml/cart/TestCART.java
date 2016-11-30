package com.spbsu.ml.cart;

import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.ColMajorArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedL2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.cart.CARTTreeOptimization;
import com.spbsu.ml.methods.cart.CARTTreeOptimizationFixError;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;


/**
 * Created by n_buga on 16.10.16.
 */


public class TestCART {
    Random r = new Random();

    double nextDouble(double range1, double range2) {
        return range1 + r.nextDouble()*(range2 - range1);
    }
/*
    @Test
    public void TestSimple1Dim() { // i -> i^2
        final CARTTreeOptimization testObj = new CARTTreeOptimization();
        int k = 10;
        double[] data = new double[k];
        Vec target = new ArrayVec(k);
        for (int i = 0; i < k; i++) {
            data[i] = i;
            target.set(i, i*i);
        }

        Mx mx = new ColMajorArrayMx(k, data);
        VecDataSet learn = new VecDataSetImpl(mx, null);
        L2 func = new L2(target, learn);
        Trans f = testObj.fit(learn, func);

        double disp = 0;
        double[] quest = new double[k];
        for (int i = 0; i < k/2; i++) {
            quest[i] = i + 0.5;
            double ans = f.compute(new ArrayVec(quest, i, 1)).get(0);
            disp += Math.pow((ans - quest[i]*quest[i]), 2);
        }

        for (int i = k/2; i < k; i++) {
            quest[i] = i;
            double ans = f.compute(new ArrayVec(quest, i, 1)).get(0);
            disp += Math.pow((ans - quest[i]*quest[i]), 2);
        }

        System.out.println(disp/k);

*/
/*        for (int i = 0; i < k; i++) {
            System.out.printf("%f ", quest[i]);
            System.out.println(f.compute(new ArrayVec(quest, i, 1)).get(0));
        } *//*

    }

    @Test
    public void TestSimple2Dim() {
        final CARTTreeOptimization testObj = new CARTTreeOptimization(); // i^2, sqrt i -> i
        int k = 10;
        double[] data = new double[2*k];
        Vec target = new ArrayVec(k);
        for (int i = 0; i < k; i++) {
            target.set(i, i);
            data[i] = i*i;
            data[k + i] = Math.sqrt(i);
        }

        Mx mx = new ColMajorArrayMx(k, data);
        VecDataSet learn = new VecDataSetImpl(mx, null);
        L2 func = new L2(target, learn);
        Trans f = testObj.fit(learn, func);

        double disp = 0;
        double quest[] = new double[2*k];
        for (int i = 0; i < k; i++) {
            Random r = new Random();
            double a = r.nextDouble()*k;
            quest[2*i] = a*a;
            quest[2*i + 1] = Math.sqrt(a);
            double ans = f.compute(new ArrayVec(quest, 2*i, 2)).get(0);
            disp += Math.pow((ans - a), 2);
        }

        System.out.println(disp/k);

*/
/*        for (int i = 0; i < k; i++) {
            System.out.printf("%f %f", quest[2*i], quest[2*i + 1]);
            System.out.println(f.compute(new ArrayVec(quest, 2*i, 2)));
        } *//*

    }

*/
/*    @Test
    public void testRandom2Dim() {
        CARTTreeOptimization testObj = new CARTTreeOptimization();
        int k = 10;
        double[] data = new double[2*k];
        Vec target = new ArrayVec(k);
        int max_bound = 50;
        int min_bound = -50;
        for (int i = 0; i < k; i++) {
            target.set(i, nextDouble(min_bound, max_bound));
            data[i] = nextDouble(min_bound, max_bound);
            data[k + i] = nextDouble(min_bound, max_bound);
        }

        Mx mx = new ColMajorArrayMx(k, data);
        VecDataSet learn = new VecDataSetImpl(mx, null);
        L2 func = new L2(target, learn);
        Trans f = testObj.fit(learn, func);

        double quest[] = new double[2*k];
        for (int i = 0; i < k; i++) {
            Random r = new Random();
            double a = r.nextDouble()*k;
            quest[2*i] = a;
            quest[2*i + 1] = a*a;
        }

        for (int i = 0; i < k; i++) {
            System.out.printf("%f %f", quest[2*i], quest[2*i + 1]);
            System.out.println(f.compute(new ArrayVec(quest, 2*i, 2)));
        }
    } */


    @Test
    public void testnDim() { //function majority
        int n = 3;
        int k = 10;
        Vec data[] = new Vec[k];
        Vec target = new ArrayVec(k);

        data[0] = new ArrayVec(new double[] {0, 1, 0});
        data[1] = new ArrayVec(new double[] {1, 1, 1});
        data[2] = new ArrayVec(new double[] {0, 0, 1});
        data[3] = new ArrayVec(new double[] {0, 0, 0});
        data[4] = new ArrayVec(new double[] {1, 1, 1});
        data[5] = new ArrayVec(new double[] {0 ,0, 0});
        data[6] = new ArrayVec(new double[] {1, 0, 1});
        data[7] = new ArrayVec(new double[] {0, 1, 0});
        data[8] = new ArrayVec(new double[] {0, 0, 0});
        data[9] = new ArrayVec(new double[] {0, 1, 1});

        for (int i = 0; i < k; i++) {
            int sum_1 = 0;
            for (int j = 0; j < n; j++) {
                sum_1 += data[i].get(j);
            }
            if (sum_1 > n/2) {
                target.set(i, 1);
            } else {
                target.set(i, 0);
            }
        }

        Mx mx = new RowsVecArrayMx(data);
        VecDataSet learn = new VecDataSetImpl(mx, null);

        final CARTTreeOptimizationFixError testObj = new CARTTreeOptimizationFixError(learn);

        int weights[] = new int[learn.length()];
        Arrays.fill(weights, 1);
        WeightedLoss func = new WeightedLoss(new L2(target, learn), weights);
        Trans f = testObj.fit(learn, func);

        double disp = 0;
        double quest[] = new double[n*k];
        for (int i = 0; i < k; i++) {
            int sum1 = 0;
            for (int j = 0; j < n; j++) {
                quest[i*n + j] = r.nextInt(2);
                sum1 += quest[i*n + j];
            }
            double ans = f.compute(new ArrayVec(quest, i * n, n)).get(0);
            double real_ans = 0;
            if (sum1 > n/2) {
                real_ans = 1;
            }
            disp += Math.pow((ans - real_ans), 2);
        }

        System.out.println(disp/k);
    }

    @Test
    public void testnDimRand() { //function majority
        int n = 10;
        int k = 201;
        Vec data[] = new Vec[k];
        Vec target = new ArrayVec(k);

        for (int i = 0; i < k; i++) {
            int sum1 = 0;
            data[i] = new ArrayVec(n);
            for (int j = 0; j < n; j++) {
                data[i].set(j, r.nextInt(2));
                sum1 += data[i].get(j);
            }
            if (sum1 > n/2) {
                target.set(i, 1);
            } else {
                target.set(i, 0);
            }
        }

        Mx mx = new RowsVecArrayMx(data);
        VecDataSet learn = new VecDataSetImpl(mx, null);

        final CARTTreeOptimization testObj = new CARTTreeOptimization(learn);

        int weights[] = new int[learn.length()];
        Arrays.fill(weights, 1);
        WeightedLoss func = new WeightedLoss(new L2(target, learn), weights);
        Trans f = testObj.fit(learn, func);

        double disp = 0;
        double quest[] = new double[n*k];
        for (int i = 0; i < k; i++) {
            int cnt = 0;
            for (int j = 0; j < n; j++) {
                quest[i*n + j] = r.nextInt(2);
                cnt += quest[i*n + j];
            }
            double ans = f.compute(new ArrayVec(quest, i*n, n)).get(0);
            int real_ans;
            if (cnt > n/2) {
                real_ans = 1;
            } else {
                real_ans = 0;
            }
            disp += Math.pow((real_ans - ans), 2);
        }

        disp /= k;

        System.out.println(disp);

        for (int i = 0; i < k; i++) {
            double ans = f.compute(data[i]).get(0);
            double right_ans = target.get(i);
//            assert(Math.abs(ans - right_ans) <= testObj.getMaxError());
        }
    }
}

