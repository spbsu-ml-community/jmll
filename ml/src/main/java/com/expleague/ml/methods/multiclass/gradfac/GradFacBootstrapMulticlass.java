package com.expleague.ml.methods.multiclass.gradfac;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.commons.util.Pair;
import com.expleague.commons.math.Func;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.func.ScaledFunc;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 25.12.14
 */
public class GradFacBootstrapMulticlass implements VecOptimization<L2> {
  private final VecOptimization inner;
  private final Factorization matrixDecomposition;
  private final Class<? extends L2> local;
  private final boolean printErrors;

  public GradFacBootstrapMulticlass(final VecOptimization<L2> inner, final Factorization matrixDecomposition, final Class<? extends L2> local) {
    this(inner, matrixDecomposition, local, false);
  }

  public GradFacBootstrapMulticlass(final VecOptimization<L2> inner, final Factorization matrixDecomposition, final Class<? extends L2> local, final boolean printErrors) {
    this.inner = inner;
    this.matrixDecomposition = matrixDecomposition;
    this.local = local;
    this.printErrors = printErrors;
  }

  @Override
  public FuncJoin fit(VecDataSet learn, L2 mllLogitGradient) {
    final Mx gradient = (Mx)mllLogitGradient.target;
    final Pair<Vec, Vec> pair = matrixDecomposition.factorize(gradient);

    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();

    final double normB = VecTools.norm(b);
    VecTools.scale(b, 1 / normB);
    VecTools.scale(h, normB);

    final L2 loss = DataTools.newTarget(local, h, learn);
    final WeightedLoss weightedLoss = createBootstrapedTarget(gradient, VecTools.outer(h, b), loss);
    final Func model = (Func) inner.fit(learn, weightedLoss);

    final Func[] models = new Func[gradient.columns()];
    for (int c = 0; c < models.length; c++) {
      models[c] = new ScaledFunc(b.get(c), model);
    }
    final FuncJoin resultModel = new FuncJoin(models);

    if (printErrors) {
      final Mx mxAfterFactor = VecTools.outer(h, b);
      final Mx mxAfterFit = resultModel.transAll(learn.data());
      final double error1 = VecTools.distance(gradient, mxAfterFactor);
      final double error2 = VecTools.distance(mxAfterFactor, mxAfterFit);
      final double totalError = VecTools.distance(gradient, mxAfterFit);
      System.out.println(String.format("err1 = %f, err2 = %f, absErr = %f", error1, error2, totalError));
    }

    return resultModel; //not MultiClassModel, for boosting compatibility
  }

  private static WeightedLoss createBootstrapedTarget(final Mx gradMx, final Mx approxMx, final L2 lossForFit) {
    if (gradMx.rows() != approxMx.rows() || gradMx.columns() != approxMx.columns()) {
      throw new IllegalArgumentException("What the fuck with dimensions?");
    }

    final float[] weights = new float[lossForFit.dim()];
    for (int i = 0; i < gradMx.rows(); i++) {
      final double error = VecTools.l1(VecTools.subtract(gradMx.row(i), approxMx.row(i)));
      weights[i] = (float) (gradMx.columns() / error);
    }
    return new WeightedLoss<>(lossForFit, weights);
  }
}
