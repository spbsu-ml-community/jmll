package com.spbsu.ml.models.multiclass;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;

/**
 * User: qdeee
 * Date: 14.04.14
 * Description:
 */
public interface MCModel extends Func {
  int countClasses();
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
