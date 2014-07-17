package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.models.MultiClassModel;

/**
 * User: solar
 * Date: 27.11.13
 * Time: 18:55
 */
public class MultiClass extends VecOptimization.Stub<L2> {
  private final VecOptimization<L2> inner;
  private final Class<? extends L2> local;

  public MultiClass(VecOptimization<L2> inner, Class<? extends L2> local) {
    this.inner = inner;
    this.local = local;
  }

  @Override
  public MultiClassModel fit(VecDataSet learn, L2 mllLogitGradient) {
    final Mx data = learn.data();
    final Mx gradient = new VecBasedMx(data.rows(), mllLogitGradient.target);
    final Func[] models = new Func[gradient.rows()];
    for (int c = 0; c < models.length; c++) {
      models[c] = (Func)inner.fit(learn, DataTools.newTarget(local, gradient.row(c), learn));
    }
    return new MultiClassModel(models);
  }
}
