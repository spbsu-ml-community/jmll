package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.optimization.FuncConvex;
import com.spbsu.ml.optimization.Optimize;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 10.12.13
 * Time: 20:48
 * Idea please stop making my code yellow
 */
@Deprecated
public class MagicDescent implements Optimize<FuncConvex> {
    private static final Logger LOG = Logger.create(GradientDescent.class);
    private final Vec x0;
    private final double eps;

    public MagicDescent(final Vec x0, final double eps) {
        this.x0 = x0;
        this.eps = eps;
    }

    Vec addXtoCY(final Vec x, final Vec y, final double c) {
        final Vec ans = copy(x);
        for (int i = 0; i < x.dim(); i++)
            ans.set(i, x.get(i) + y.get(i) * c);
        return ans;
    }

    double scaleMultiply(final Vec x, final Vec y) {
        double ans = 0;
        for (int i = 0; i < x.dim(); i++)
            ans += x.get(i) * y.get(i);
        return ans;
    }

    @Override
    public Vec optimize(final FuncConvex func) {
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
