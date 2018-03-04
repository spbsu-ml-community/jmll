package com.expleague.ml.distributions;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

/**
 * Created by noxoomo on 11/02/2018.
 */
public class VecSufficientStat implements SufficientStatistic {
  Vec stats;

  VecSufficientStat(int dim) {
    stats = new ArrayVec(dim);
  }

  @Override
  public SufficientStatistic combine(SufficientStatistic other) {
    if (other instanceof VecSufficientStat) {
      stats = VecTools.append(stats, ((VecSufficientStat) other).stats);
    }
    else {
      throw new IllegalArgumentException("wrong stat class");
    }
    return this;
  }

  public Vec stats() {
    return stats;
  }

}
