package com.expleague.exp.multiclass.weak;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.math.Trans;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.MultiClassModel;
import gnu.trove.map.hash.TIntIntHashMap;

/**
* User: qdeee
* Date: 16.08.14
*/
public class CustomWeakBinClass extends VecOptimization.Stub<LLLogit> {
  private final int iters;
  private final double step;

  public CustomWeakBinClass(final int iters, final double step) {
    this.iters = iters;
    this.step = step;
  }

  @Override
  public Trans fit(final VecDataSet learn, final LLLogit targetFunc) {
    final Vec binClassTarget = targetFunc.labels();
    final IntSeq intBinClassTarget = VecTools.toIntSeq(binClassTarget);
    final IntSeq mcTarget = MCTools.normalizeTarget(intBinClassTarget, new TIntIntHashMap());

    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(iters, step);
    final MultiClassModel mcm = (MultiClassModel) customWeakMultiClass.fit(learn, new BlockwiseMLLLogit(mcTarget, learn));
    return new Func.Stub() {
      @Override
      public double value(Vec x) {
        return mcm.getInternModel().trans(x).get(0);
      }

      @Override
      public int dim() {
        return mcm.getInternModel().xdim();
      }
    };
  }
}
