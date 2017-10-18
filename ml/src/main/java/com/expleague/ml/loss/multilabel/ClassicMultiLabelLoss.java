package com.expleague.ml.loss.multilabel;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.ml.TargetFunc;

/**
 * User: qdeee
 * Date: 20.03.15
 */
public interface ClassicMultiLabelLoss extends Func, TargetFunc{
  Mx getTargets();
}
