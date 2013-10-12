package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Model;
import com.spbsu.ml.data.DataTools;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 01.03.11
 * Time: 22:30
 */
public class NormalizedLinearModel extends Model {
    private final Vec weights;
    DataTools.NormalizationProperties props;

    public NormalizedLinearModel(Vec weights, DataTools.NormalizationProperties props) {
        this.weights = weights;
        this.props = props;
    }

    public double value(Vec point) {
        Vec x = copy(point);
        append(x, props.xMean);
        x = multiply(props.xTrans, x);
        return multiply(weights, x) / props.yVar + props.yMean;
    }
}
