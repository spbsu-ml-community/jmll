package com.spbsu.ml.loss.multilabel;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.TargetFunc;

/**
 * User: qdeee
 * Date: 20.03.15
 */
public interface ClassicMultiLabelLoss extends Func, TargetFunc{
  Mx getTargets();
}
