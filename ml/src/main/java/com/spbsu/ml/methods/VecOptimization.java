package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.TargetFunc;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface VecOptimization<Loss extends TargetFunc> extends Optimization<Loss, VecDataSet, Vec> {
  /**
   * Optimization based on vector representation of train items
   */
  @Override
  Trans fit(VecDataSet learn, Loss loss);

  abstract class Stub<Loss extends TargetFunc> implements VecOptimization<Loss> {
  }
}
