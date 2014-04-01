package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.optimization.Optimize;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Random;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public class StochasticGradientDescent implements Optimize<FuncEnsemble<FuncC1>> {
  private final int iterations;
  private final double step;
  private Random rand;

  public StochasticGradientDescent(final int iterations, final double step, Random rand) {
    this.iterations = iterations;
    this.step = step;
    this.rand = rand;
  }

  @Override
  public Vec optimize(final FuncEnsemble<FuncC1> sumFuncs) {
    Vec x = new ArrayVec(sumFuncs.xdim());
    for (int i = 0; i < x.dim(); i++) {
      x.set(i, rand.nextGaussian());
    }

    int iter = 0;
    TIntList indices = new TIntArrayList(ArrayTools.sequence(0, sumFuncs.size()));
    while (iter++ < iterations) {
      indices.shuffle(rand);
      for (int i = 0; i < indices.size(); i++) {
        VecTools.incscale(x, sumFuncs.models[i].gradient(x), -step);
      }
    }
    return x;
  }
}
