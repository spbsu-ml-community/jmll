package com.expleague.ml.methods;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.TargetFunc;

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
