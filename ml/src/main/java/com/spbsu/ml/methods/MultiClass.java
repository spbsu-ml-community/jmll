package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
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
  private final boolean printErrors;

  public MultiClass(final VecOptimization<L2> inner, final Class<? extends L2> local) {
    this(inner, local, false);
  }

  public MultiClass(final VecOptimization<L2> inner, final Class<? extends L2> local, final boolean printErrors) {
    this.inner = inner;
    this.local = local;
    this.printErrors = printErrors;
  }

  @Override
  public FuncJoin fit(final VecDataSet learn, final L2 mllLogitGradient) {
    final Mx gradient;
    if (mllLogitGradient.target instanceof Mx) {
      gradient = (Mx)mllLogitGradient.target;
    } else {
      final int columns = mllLogitGradient.target.dim() / learn.data().rows();
      gradient = new VecBasedMx(columns, mllLogitGradient.target);
    }
    final Func[] models = new Func[gradient.columns()];
    for (int c = 0; c < models.length; c++) {
      final L2 loss = DataTools.newTarget(local, gradient.col(c), learn);
      models[c] = (Func)inner.fit(learn, loss);
    }
    final FuncJoin resultModel = new FuncJoin(models);

    if (printErrors) {
      final Mx mxAfterFit = resultModel.transAll(learn.data());
      final double error = VecTools.distance(gradient, mxAfterFit);
      final double gradNorm = VecTools.norm(gradient);
      System.out.println("grad_norm = " + gradNorm + ", err = " + error);
    }

    return resultModel; //not MultiClassModel, for boosting compatibility
  }
}
