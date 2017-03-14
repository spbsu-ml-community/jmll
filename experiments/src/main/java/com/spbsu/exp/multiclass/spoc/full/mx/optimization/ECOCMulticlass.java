package com.spbsu.exp.multiclass.spoc.full.mx.optimization;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

/**
 * User: qdeee
 * Date: 21.11.14
 */
public class ECOCMulticlass extends VecOptimization.Stub<L2> {
  private final VecOptimization<L2> inner;
  private final Class<? extends L2> localLossClass;
  private final int K;
  private final int L;
  private final double stepB;

  public ECOCMulticlass(final VecOptimization<L2> inner, final Class<? extends L2> localLossClass, final int k, final int l, final double stepB) {
    this.inner = inner;
    this.localLossClass = localLossClass;
    this.K = k;
    this.L = l;
    this.stepB = stepB;
  }

  @Override
  public Trans fit(final VecDataSet learn, final L2 smllGradient) {
    final Mx grad = (Mx) smllGradient.target;
    final Mx gradB = grad.sub(0, 0, K - 1, L);
    final Mx gradF = grad.sub(K - 1, 0, learn.length(), L);

    final Mx newB = VecTools.scale(gradB, stepB);

    final Func[] classifiers = new Func[grad.columns()];
    for (int j = 0; j < grad.columns(); j++) {
      final L2 localLoss = DataTools.newTarget(localLossClass, gradF.col(j), learn);
      classifiers[j] = (Func) inner.fit(learn, localLoss);
    }
    return new BoostingCompatibleFuncJoin(new FuncJoin(classifiers), newB);
  }

  public static class BoostingCompatibleFuncJoin extends FuncJoin {
    private Mx B;

    public BoostingCompatibleFuncJoin(final FuncJoin funcJoin, final Mx B) {
      super(funcJoin.dirs());
      this.B = B;
    }

    public Mx getB() {
      return B;
    }

    @Override
    public Mx transAll(final Mx ds) {
      final Mx result = new VecBasedMx(B.rows() + ds.rows(), B.columns());
      final Mx resultB = result.sub(0, 0, B.rows(), B.columns());
      VecTools.assign(resultB, B);

      final Mx resultF = result.sub(B.rows(), 0, ds.rows(), B.columns());
      final Mx evalFPart = super.transAll(ds);
      VecTools.assign(resultF, evalFPart);

      return result;
    }
  }
}
