package com.spbsu.ml.methods.multiclass.gradfac;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.factorization.impl.SVDAdapterEjml;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 25.12.14
 */
public class GradFacSvdNMulticlass implements VecOptimization<L2> {
  private final VecOptimization<L2> inner;
  private final Class<? extends L2> local;
  private final SVDAdapterEjml outerFactorization;
  private final boolean printErrors;

  public GradFacSvdNMulticlass(final VecOptimization<L2> inner, final Class<? extends L2> local) {
    this(inner, local, 1);
  }

  public GradFacSvdNMulticlass(final VecOptimization<L2> inner, final Class<? extends L2> local, final int factorDim) {
    this(inner, local, factorDim, true, false);
  }

  public GradFacSvdNMulticlass(final VecOptimization<L2> inner, final Class<? extends L2> local, final int factorDim, final boolean needCompact, final boolean printErrors) {
    this.inner = inner;
    this.local = local;
    this.printErrors = printErrors;
    this.outerFactorization = new SVDAdapterEjml(factorDim, needCompact);
  }

  @Override
  public FuncJoin fit(VecDataSet learn, L2 mllLogitGradient) {
    final Mx gradient = mllLogitGradient.target instanceof Mx
        ? (Mx)mllLogitGradient.target
        : new VecBasedMx(mllLogitGradient.target.dim() / learn.length(), mllLogitGradient.target);

    final Pair<Vec, Vec> pair = outerFactorization.factorize(gradient);

    final Mx h = (Mx) pair.getFirst();
    final Mx b = (Mx) pair.getSecond();

    final Func[] uApproxModels = new Func[b.columns()];

    for (int j = 0; j < b.columns(); j++) {
      final L2 loss = DataTools.newTarget(local, h.col(j), learn);
      uApproxModels[j] = (Func) inner.fit(learn, loss);
    }

    final Func[] models = new Func[gradient.columns()];
    for (int c = 0; c < models.length; c++) {
      models[c] = new FuncEnsemble<>(uApproxModels, b.row(c));
    }
    final FuncJoin resultModel = new FuncJoin(models);

    if (printErrors) {
      final Mx mxAfterFactor = MxTools.multiply(h, MxTools.transpose(b));
      final Mx mxAfterFit = resultModel.transAll(learn.data());
      final double gradNorm = VecTools.norm(gradient);
      final double error1 = VecTools.distance(gradient, mxAfterFactor);
      final double error2 = VecTools.distance(mxAfterFactor, mxAfterFit);
      final double totalError = VecTools.distance(gradient, mxAfterFit);

      System.out.println(String.format("grad_norm = %f, err1 = %f, err2 = %f, absErr = %f", gradNorm, error1, error2, totalError));
    }

    return resultModel; //not MultiClassModel, for boosting compatibility
  }
}
