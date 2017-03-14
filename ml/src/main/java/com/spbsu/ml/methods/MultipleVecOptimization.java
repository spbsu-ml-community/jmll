package com.spbsu.ml.methods;

import com.spbsu.ml.TargetFunc;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;

import java.lang.reflect.Array;

/**
 * User: noxoomo
 */

public interface MultipleVecOptimization<Loss extends TargetFunc>  extends VecOptimization<Loss> {
  Trans[] fit(VecDataSet[] learn, Loss[] loss);

  abstract class Stub<Loss extends TargetFunc> implements MultipleVecOptimization<Loss> {
    @Override
    public Trans fit(VecDataSet learn, Loss loss) {
      VecDataSet[] ds = new VecDataSet[1];
      Loss[] losses = (Loss[]) Array.newInstance(loss.getClass(),1);
      ds[0] = learn;
      losses[0] = loss;
      return fit(ds, losses)[0];
    }
  }
}
