package com.spbsu.ml.func;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Trans;

/**
 * User: solar
 * Date: 26.11.12
 * Time: 15:56
 */
public class Average<F extends Trans> extends Trans.Stub {
  public final F[] models;

  public Average(F[] models) {
    this.models = models;
  }

  @Override
  public int xdim() {
    return models[0].xdim() * models.length;
  }

  @Override
  public int ydim() {
    return models[ArrayTools.max(models, new Evaluator<F>() {
      @Override
      public double value(F f) {
        return f.ydim();
      }
    })].ydim();
  }

  @Override
  public Trans gradient() {
    return new Average<Trans>(ArrayTools.map(models, Trans.class, new Computable<F, Trans>() {
      @Override
      public Trans compute(F argument) {
        return argument.gradient();
      }
    }));
  }

  @Override
  public Vec trans(Vec x) {
    Vec result = new ArrayVec(ydim());
    for (int i = 0; i < models.length; i++) {
      VecTools.append(result, models[i].trans(x));
    }
    VecTools.scale(result, 1./models.length);
    return result;
  }
}
