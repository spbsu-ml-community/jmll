package com.spbsu.ml.data.tools;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.seq.Seq;

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
}
