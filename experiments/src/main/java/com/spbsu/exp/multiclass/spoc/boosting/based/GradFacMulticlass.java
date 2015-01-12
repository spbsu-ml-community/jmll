package com.spbsu.exp.multiclass.spoc.boosting.based;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.factorization.OuterFactorization;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.func.ScaledFunc;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 25.12.14
 */
public class GradFacMulticlass implements VecOptimization<L2> {
  private final VecOptimization<L2> inner;
  private final OuterFactorization matrixDecomposition;
  private final Class<? extends L2> local;

  public GradFacMulticlass(final VecOptimization<L2> inner, final OuterFactorization matrixDecomposition, final Class<? extends L2> local) {
    this.inner = inner;
    this.matrixDecomposition = matrixDecomposition;
    this.local = local;
  }

  @Override
  public FuncJoin fit(VecDataSet learn, L2 mllLogitGradient) {
    final Mx gradient = (Mx)mllLogitGradient.target;
    final Pair<Vec, Vec> pair = matrixDecomposition.factorize(gradient);

    final Vec h = pair.getFirst();
    final Vec b = pair.getSecond();

    System.out.println("||h|| = " + VecTools.norm(h) + ", ||b|| = " + VecTools.norm(b) + ", RMSE = " + VecTools.distance(gradient, VecTools.outer(h, b)));

    final L2 loss = DataTools.newTarget(local, h, learn);
    final Func model = (Func) inner.fit(learn, loss);

    final Func[] models = new Func[gradient.columns()];
    for (int c = 0; c < models.length; c++) {
      models[c] = new ScaledFunc(b.get(c), model);
    }
    return new FuncJoin(models); //not MultiClassModel, for boosting compatibility
  }
}
