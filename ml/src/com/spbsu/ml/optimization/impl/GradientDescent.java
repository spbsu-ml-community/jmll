package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.optimization.ConvexFunction;
import com.spbsu.ml.optimization.ConvexOptimize;
import com.spbsu.ml.optimization.QuadraticFunction;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: qde
 * Date: 25.04.13
 * Time: 23:41
 */
public class GradientDescent implements ConvexOptimize {
    private static Logger LOG = Logger.create(GradientDescent.class);
    private Vec x0;

    public GradientDescent(Vec x0) {
        this.x0 = x0;
    }

    @Override
    public Vec optimize(ConvexFunction func, double eps) {
        boolean isQuadraticFunc = func instanceof QuadraticFunction;

        double constStep = 1.0 / func.getGradLipParam();

        Vec x1 = copy(x0);
        Vec x2 = new ArrayVec(x0.dim());
        Vec grad = func.gradient().value(x0);

        int iter = 0;

        double distance = 1;
        while (distance > eps && iter < 5000000) {
            double step = isQuadraticFunc? getStepSizeForQuadraticFunc(func, grad) : constStep;
            for (int i = 0; i < x2.dim(); i++) {
                x2.set(i, x1.get(i) - grad.get(i) * step);
            }

            x1 = copy(x2);
            grad = func.gradient().value(x1);
            distance = VecTools.norm(grad) / func.getGlobalConvexParam();
            iter++;
        }

        LOG.message("GDM iterations = " + iter + "\n\n");
        return x2;
    }

    private double getStepSizeForQuadraticFunc(ConvexFunction func, Vec grad) {
            return VecTools.multiply(grad, grad) / ((QuadraticFunction) func).getQuadrPartValue(grad);
    }
}
