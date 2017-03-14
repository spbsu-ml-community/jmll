package com.spbsu.exp.multiclass.weak;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.MultiClassModel;
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
    return mcm.getInternModel().dirs()[0];
  }
}
