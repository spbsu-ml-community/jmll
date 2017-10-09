package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

public abstract class ParameterOptimization<Loss extends TargetFunc, DSType extends DataSet<DSItem>, DSItem> implements Optimization<Loss,DSType,DSItem> {
  private final Computable<Vec, Computable<DSItem, Vec>> factory;

  protected ParameterOptimization(Computable<Vec, Computable<DSItem, Vec>> factory) {
    this.factory = factory;
  }

  @Override
  public final Computable<DSItem, Vec> fit(DSType learn, Loss loss) {
    return factory.compute(fitVec(learn, loss));
  }

  protected abstract Vec fitVec(DSType learn, Loss loss);
}
