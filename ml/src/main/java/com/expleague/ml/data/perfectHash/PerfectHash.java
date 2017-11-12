package com.expleague.ml.data.perfectHash;

import com.expleague.commons.func.Action;
import com.expleague.commons.func.WeakListenerHolder;
import com.expleague.ml.distributions.parametric.BetaDistribution;
import gnu.trove.map.hash.TDoubleIntHashMap;

public interface PerfectHash<U> extends WeakListenerHolder<Integer> {
  int id(U value);
  int size();

}


