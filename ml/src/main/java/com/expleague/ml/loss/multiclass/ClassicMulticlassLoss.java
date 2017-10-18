package com.expleague.ml.loss.multiclass;

import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.math.Func;
import com.expleague.ml.TargetFunc;

/**
 * User: qdeee
 * Date: 22.01.15
 */
public interface ClassicMulticlassLoss extends Func, TargetFunc {
  IntSeq labels();
}
