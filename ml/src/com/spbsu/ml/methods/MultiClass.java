package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.L2;

/**
 * User: solar
 * Date: 27.11.13
 * Time: 18:55
 */
public class MultiClass implements Optimization<L2> {
  private final Optimization<L2> inner;
  private final Computable<Vec, ? extends L2> local;

  public MultiClass(Optimization<L2> inner, Computable<Vec, ? extends L2> local) {
    this.inner = inner;
    this.local = local;
  }

  @Override
  public FuncJoin fit(DataSet learn, L2 mllLogitGradient) {
    final Mx data = learn.data();
    final Mx gradient = new VecBasedMx(data.rows(), mllLogitGradient.target);
    final Func[] models = new Func[gradient.rows()];
    for (int c = 0; c < models.length; c++) {
      models[c] = (Func)inner.fit(learn, local.compute(gradient.row(c)));
    }
    return new FuncJoin(models);
  }
}
