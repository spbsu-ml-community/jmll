package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.optimization.Optimize;
import com.spbsu.ml.optimization.TensorNetFunction;

/**
 * User: qdeee
 * Date: 09.09.13
 */
public class ALS implements Optimize<TensorNetFunction>{
    private Vec x0;
    private int iterCount;

    public ALS(Vec x0, int iterCount) {
        this.x0 = x0;
        this.iterCount = iterCount;
    }

    @Override
    public Vec optimize(TensorNetFunction func) {
        int n = func.xdim();

        Vec u = new ArrayVec(n);
        Vec v = new ArrayVec(n);

        for (int i = 0; i < n; i++) {
            u.set(i, x0.get(i));
            v.set(i, x0.get(i+n));
        }

        int iter = 0;
        while (iter++ < iterCount) {
            double squareNormV = Math.pow(VecTools.norm(v), 2);
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    sum += v.get(j) * func.getX().get(i, j);
                }
                u.set(i, sum / squareNormV);
            }

            double squareNormU = Math.pow(VecTools.norm(u), 2);
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    sum += u.get(i) * func.getX().get(i, j);
                }
                v.set(j, sum / squareNormU);
            }
        }
        Vec result = new ArrayVec(n * 2);
        for (int i = 0; i < n; i++) {
            result.set(i, u.get(i));
            result.set(i + n, v.get(i));
        }
        return result;
    }
}
