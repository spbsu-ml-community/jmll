package com.expleague.ml.optimization;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.ReguralizerFunc;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.io.IOException;
import java.util.Random;

/**
 * User: qdeee
 * Date: 17.03.14
 */
public class StochasticGradientDescent implements Optimize<FuncEnsemble<? extends FuncC1>> {
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
  public Vec optimize(FuncEnsemble sumFuncs, ReguralizerFunc reg, Vec x0) {
    if (sumFuncs.last().gradient() == null) {
      throw new IllegalArgumentException("Internal functions must implement not-null gradient()");
    }

    x = x0;
    final String anim = "|/-\\";

    int iter = 0;
    final TIntList indices = new TIntArrayList(ArrayTools.sequence(0, sumFuncs.size()));
    while (iter++ < iterations) {
      indices.shuffle(rand);
      for (int i = 0; i < indices.size(); i++) {
        final FuncC1 func = (FuncC1) sumFuncs.models[indices.get(i)];
        Vec gradient = func.gradient(x);
        VecTools.incscale(x, gradient, -step);

        if (i % 100 == 0) {
          final double percent = (double) i / indices.size();
          final int percentIdx = (int) (percent * 1000);
          final String data = "\r" + anim.charAt(percentIdx % anim.length()) + " " + String.format("%.2f", percent) + "%"
               + "\t f(x) = " + func.value(x) + "\t |x| = " + VecTools.norm(gradient);
          try {
            System.out.write(data.getBytes());
          }
          catch (IOException e) {
            throw new RuntimeException(e);
          }
        }
      }
//      System.out.println(x);
      System.out.println("\nepoch [" + iter + "] = " + sumFuncs.value(x) / step / sumFuncs.dim());
    }
    return x;
  }
}
