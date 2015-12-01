package com.spbsu.ml.methods.multiclass.gradfac;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.factorization.OuterFactorization;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.func.ScaledFunc;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multiclass.MultiClassOneVsRest;

/**
 * User: qdeee
 * Date: 25.12.14
 */
public class GradFacMulticlass implements VecOptimization<L2> {
  private final VecOptimization<L2> inner;
  private final OuterFactorization matrixDecomposition;
  private final Class<? extends L2> local;
  private final boolean printErrors;

  public GradFacMulticlass(final VecOptimization<L2> inner, final OuterFactorization matrixDecomposition, final Class<? extends L2> local) {
    this(inner, matrixDecomposition, local, false);
  }

  public GradFacMulticlass(final VecOptimization<L2> inner, final OuterFactorization matrixDecomposition, final Class<? extends L2> local, final boolean printErrors) {
    this.inner = inner;
    this.matrixDecomposition = matrixDecomposition;
    this.local = local;
    this.printErrors = printErrors;
  }

  @Override
  public FuncJoin fit(VecDataSet learn, L2 mllLogitGradient) {
    final Mx gradient = mllLogitGradient.target instanceof Mx
        ? (Mx)mllLogitGradient.target
        : new VecBasedMx(mllLogitGradient.target.dim() / learn.length(), mllLogitGradient.target);
    final Pair<Vec, Vec> pair = matrixDecomposition.factorize(gradient);

    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();

    final double normB = VecTools.norm(b);
    VecTools.scale(b, 1 / normB);
    VecTools.scale(h, normB);

    final L2 loss = DataTools.newTarget(local, h, learn);
    final Func model = MultiClassOneVsRest.extractFunc(inner.fit(learn, loss));

    final Func[] models = new Func[gradient.columns()];
    for (int c = 0; c < models.length; c++) {
      models[c] = new ScaledFunc(b.get(c), model);
    }
    final FuncJoin resultModel = new FuncJoin(models);

    if (printErrors) {
      final Mx mxAfterFactor = VecTools.outer(h, b);
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
