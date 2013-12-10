package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.optimization.ConvexFunction;
import com.spbsu.ml.optimization.ConvexOptimize;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 10.12.13
 * Time: 20:48
 * Idea please stop making my code yellow
 */
@Deprecated
public class MagicDescent implements ConvexOptimize {
    private static Logger LOG = Logger.create(GradientDescent.class);
    private Vec x0;

    public MagicDescent(Vec x0) {
        this.x0 = x0;
    }

    Vec addXtoCY(Vec x, Vec y, double c) {
        Vec ans = copy(x);
        for (int i = 0; i < x.dim(); i++)
            ans.set(i, x.get(i) + y.get(i) * c);
        return ans;
    }

    double scaleMultiply(Vec x, Vec y) {
        double ans = 0;
        for (int i = 0; i < x.dim(); i++)
            ans += x.get(i) * y.get(i);
        return ans;
    }

    @Override
    public Vec optimize(ConvexFunction func, double eps) {
        Vec x1 = copy(x0);
        Vec grad = func.gradient().trans(x0);
        double distance = 1;
        double step = 1;
        int iter = 0;
        while (distance > eps && iter < 5000000) {
            iter++;
            while (scaleMultiply(func.gradient().trans(addXtoCY(x1, grad, -step)), grad) >= 0) {
                //System.out.println(scaleMultiply(func.gradient().trans(addXtoCY(x1, grad, -step)), grad));
                step *= 2;
            }
            while (scaleMultiply(func.gradient().trans(addXtoCY(x1, grad, -step)), grad) < 0)
                step /= 2;
            x1 = addXtoCY(x1, grad, -step);
            grad = func.gradient().trans(x1);
            distance = VecTools.norm(grad);
            //LOG.message(String.valueOf(distance));
        }
        return x1;
    }
}
