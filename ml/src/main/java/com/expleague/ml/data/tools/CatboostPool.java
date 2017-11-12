package com.expleague.ml.data.tools;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.ctrs.CtrTarget;

import java.util.List;
import java.util.Set;

/**
* User:  noxooomo
*/
public class CatboostPool extends FakePool {
  private final Set<Integer> catColumns;

  public CatboostPool(final Mx data,
                      final Seq<?> target,
                      final Set<Integer> catColumns) {
    super(data, target);
    this.catColumns = catColumns;
  }

  public boolean isCatFactor(int factorId) {
    return catColumns.contains(factorId);
  }

  public Set<Integer> catFeatureIds() {
    return catColumns;
  }
}

