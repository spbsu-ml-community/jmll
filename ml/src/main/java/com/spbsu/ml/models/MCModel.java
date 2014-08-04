package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.Func;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: qdeee
 * Date: 14.04.14
 * Description:
 */
public interface MCModel extends Func {
  Vec probs(Vec x);
  int bestClass(Vec x);
  Vec bestClassAll(Mx x); //TODO[qdeee]: use only transAll()

  abstract class Stub extends Func.Stub implements MCModel {
    @Override
    public double value(final Vec x) {
      return (double)bestClass(x);
    }

    @Override
    public Vec bestClassAll(final Mx x) {
      return transAll(x);
    }
  }
}
