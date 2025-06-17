package com.expleague.ml.func;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.Func;

import java.util.List;

/**
 * User: qdeee
 * Date: 09.04.14
 */

public class FuncEnsemble<X extends Func> extends Ensemble<X> implements Func {
    private final double scale;
    private final double bias;
    private boolean weighting = true;

    public FuncEnsemble(final X[] models, final Vec weights, double scale, double bias) {
        super(models, weights);
        this.scale = scale;
        this.bias = bias;
    }

    public FuncEnsemble(final List<X> models, double step, double scale, double bias) {
        super(models, step);
        this.scale = scale;
        this.bias = bias;
    }

    public FuncEnsemble(final X[] models, final Vec weights) {
        this(models, weights, 1.0, 0.0);
    }

    public FuncEnsemble(final List<X> models, final double step) {
        this(models, step, 1.0, 0.0);
    }

    @Override
    public double value(final Vec x) {
        double result = 0.;
        double weightTotal = 0;
        for (int i = 0; i < size(); i++) {
            double weight = weight(i);
            result += model(i).value(x) * weight;
            weightTotal += weight;
        }

        if (weighting) {
            result /= weightTotal;
        }
        return result * scale + bias;
    }

    public void setWeighting(boolean weighting) {
        this.weighting = weighting;
    }

    @Override
    public int dim() {
        return super.xdim();
    }
}
