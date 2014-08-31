package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.L2;

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
  public FuncJoin fit(VecDataSet learn, L2 mllLogitGradient) {
    final Mx gradient = (Mx)mllLogitGradient.target;
    final Func[] models = new Func[gradient.columns()];
    for (int c = 0; c < models.length; c++) {
      final L2 loss = DataTools.newTarget(local, gradient.col(c), learn);
      models[c] = (Func)inner.fit(learn, loss);
    }
    return new FuncJoin(models); //not MultiClassModel, for boosting compatibility
  }
}
