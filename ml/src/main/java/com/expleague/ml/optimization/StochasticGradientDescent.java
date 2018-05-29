package com.expleague.ml.optimization;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.RegularizerFunc;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Random;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public class StochasticGradientDescent implements Optimize<FuncEnsemble> {
  private final int iterations;
  private final double step;
  private final Random rand;
  private Vec x;

  public StochasticGradientDescent(final int iterations, final double step, final Random rand) {
    this.iterations = iterations;
    this.step = step;
    this.rand = rand;
  }


  @Override
  public Vec optimize(final FuncEnsemble sumFuncs) {
    if (sumFuncs.last().gradient() == null) {
      throw new IllegalArgumentException("Internal functions must implement not-null gradient()");
    }

    final Vec x = new ArrayVec(sumFuncs.xdim());
    for (int i = 0; i < x.dim(); i++) {
      x.set(i, rand.nextGaussian());
    }

    return optimize(sumFuncs, x);
  }

  @Override
  public Vec optimize(FuncEnsemble sumFuncs, RegularizerFunc reg, Vec x0) {
    if (sumFuncs.last().gradient() == null) {
      throw new IllegalArgumentException("Internal functions must implement not-null gradient()");
    }

    x = x0;

    int iter = 0;
    final TIntList indices = new TIntArrayList(ArrayTools.sequence(0, sumFuncs.size()));
    while (iter++ < iterations) {
      indices.shuffle(rand);
      for (int i = 0; i < indices.size(); i++) {
        VecTools.incscale(x, sumFuncs.models[indices.get(i)].gradient().trans(x), -step);
      }
      System.out.println(x);
      System.out.println(sumFuncs.value(x) / step / sumFuncs.dim());
    }
    return x;
  }
}
