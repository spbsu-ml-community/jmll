package com.expleague.ml.models.multiclass;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;

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
  Vec bestClassAll(Mx x, boolean parallel);

  abstract class Stub extends Func.Stub implements MCModel {
    @Override
    public double value(final Vec x) {
      return (double)bestClass(x);
    }

    @Override
    public Vec bestClassAll(final Mx x) {
      return transAll(x);
    }

    @Override
    public Vec bestClassAll(final Mx x, boolean parallel) {
      throw new UnsupportedOperationException();
    }
  }
}
