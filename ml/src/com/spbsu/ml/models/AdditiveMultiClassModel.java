package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.VecFunc;

import java.util.List;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 15:52
 */
public class AdditiveMultiClassModel<T extends VecFunc> {
    public final List<T> models;
    public final double step;

    public AdditiveMultiClassModel(List<T> models, double step) {
      this.models = models;
        this.step = step;
    }

    public double value(Vec point, int classNo) {
        double result = 0;
        for (T model : models) {
            result += step * model.value(point, classNo);
        }
        return result;
    }
}
