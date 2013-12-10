package com.spbsu.ml.optimization;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Trans;
import org.jetbrains.annotations.NotNull;

/**
 * User: qde
 * Date: 24.04.13
 * Time: 19:01
 */

public interface ConvexFunction extends Trans {
    @NotNull
    double getGlobalConvexParam();

    double getGradLipParam();

    double value(Vec x);
}
