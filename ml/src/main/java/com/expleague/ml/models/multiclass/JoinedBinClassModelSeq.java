package com.expleague.ml.models.multiclass;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.ArrayTools;

import java.util.function.Function;

public class JoinedBinClassModelSeq<T> implements Function<Seq<T>,Vec> {
  protected final Function<Seq<T>, Vec>[] internalModel;

  public JoinedBinClassModelSeq(final Function<Seq<T>, Vec>[] dirs) {
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
  public Vec apply(Seq<T> x) {
    return new SingleValueVec(bestClass(x));
  }

  private Vec computeSum(final Seq<T> x) {
    Vec[] values = ArrayTools.map(internalModel, Vec.class, func -> func.apply(x));
    if (values[0].dim() != 1) {
      throw new IllegalArgumentException(); //todo is it right?
    }
    final Vec sum = new ArrayVec(internalModel.length);
    for (int i = 0; i < internalModel.length; i++) {
      sum.set(i, values[i].get(0));
    }
    return sum;
  }
}
