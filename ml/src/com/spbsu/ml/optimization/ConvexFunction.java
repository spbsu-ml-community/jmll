package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Oracle1;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:01
 */

public interface ConvexFunction extends Oracle1 {
    public double getGlobalConvexParam();
    public double getLocalConvexParam(Vec x);
    public double getGradLipParam();
    public int dim();
}
