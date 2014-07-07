package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.VectorizedRealTargetDataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface VecOptimization<Loss extends Func> extends Optimization<Loss, VectorizedRealTargetDataSet<?>, Vec> {
  /**
   * Optimization based on vector representation of train items
   */
  Trans fit(VectorizedRealTargetDataSet<?> learn, Loss loss);
}
