package com.spbsu.ml;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.data.impl.BinarizedDataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface BinOptimizedModel extends Func {
  double value(BinarizedDataSet bds, int index);
  Mx transAll(BinarizedDataSet bds);

  abstract class Stub  extends Func.Stub implements BinOptimizedModel {
    public Mx transAll(BinarizedDataSet bds) {
      final Mx result = new VecBasedMx(1,bds.bins(0).length);
      for (int i = 0; i < result.dim(); i++) {
        result.set(i, value(bds, i));
      }
      return result;
    }
  }

}
