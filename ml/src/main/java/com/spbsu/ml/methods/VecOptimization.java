package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.set.VecDataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface VecOptimization<Loss extends Func> extends Optimization<Loss, VecDataSet, Vec> {
  /**
   * Optimization based on vector representation of train items
   */
  Trans fit(VecDataSet learn, Loss loss);

  abstract class Stub<Loss extends Func> implements VecOptimization<Loss> {
    @Override
    public Class<Vec> itemClass() {
      return Vec.class;
    }
  }
}
