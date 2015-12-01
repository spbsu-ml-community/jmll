package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.TargetFunc;

/**
 * User: qdeee
 * Date: 22.01.15
 */
public interface ClassicMulticlassLoss extends Func, TargetFunc {
  IntSeq labels();
}
