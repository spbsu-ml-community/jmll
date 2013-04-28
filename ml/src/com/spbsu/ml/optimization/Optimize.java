package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 18:58
 */

public interface Optimize {
    public Vec optimize(ConvexFunction func, double eps);
}
