package com.expleague.ml.models.multiclass;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.ArrayTools;

public class JoinedBinClassModelSeq<T> implements Computable<Seq<T>,Vec> {
  protected final Computable<Seq<T>, Vec>[] internalModel;

  public JoinedBinClassModelSeq(final Computable<Seq<T>, Vec>[] dirs) {
    internalModel = dirs;
  }

  public Vec probs(final Seq<T> x) {
    final Vec sum = computeSum(x);
    final Vec probs = new ArrayVec(sum.dim());
    for (int i = 0; i < sum.dim(); i++) {
      probs.set(i, MathTools.sigmoid(sum.get(i)));
    }
    return probs;
  }

  public int bestClass(final Seq<T> x) {
    final double[] trans = computeSum(x).toArray();
    return ArrayTools.max(trans);
  }

  @Override
  public Vec compute(Seq<T> x) {
    return new SingleValueVec(bestClass(x));
  }

  private Vec computeSum(final Seq<T> x) {
    Vec[] values = ArrayTools.map(internalModel, Vec.class, func -> func.compute(x));
    final Vec sum = new ArrayVec(values[0].dim());
    VecTools.append(sum, values);
    return sum;
  }
}
