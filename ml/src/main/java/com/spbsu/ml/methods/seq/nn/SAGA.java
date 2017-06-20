package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.optimization.Optimize;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Random;

public class SAGA implements Optimize<FuncEnsemble> {
  private final int iterations;
  private final double step;
  private final Random rand;

  public SAGA(final int iterations, final double step, final Random rand) {
    this.iterations = iterations;
    this.step = step;
    this.rand = rand;
  }

  public Vec optimizeWithStartX(FuncEnsemble sumFuncs, final Vec x) {
    final TIntList indices = new TIntArrayList(ArrayTools.sequence(0, sumFuncs.size()));
    final Vec[] gradients = new Vec[sumFuncs.size()];
    final Vec totalGrad = new ArrayVec(x.dim());
    for (int i = 0; i < sumFuncs.size(); i++) {
      gradients[i] = sumFuncs.models[i].gradient().trans(x);
      VecTools.append(totalGrad, gradients[i]);
    }
    for (int iter = 0; iter < iterations; iter++) {
      indices.shuffle(rand);
      for (int i = 0; i < sumFuncs.size(); i++) {
        final int modelId = indices.get(i);
        final Vec grad = sumFuncs.models[modelId].gradient().trans(x);
        VecTools.incscale(x, grad, -step);
        VecTools.incscale(x, gradients[modelId], -1 * -step);
        VecTools.incscale(x, totalGrad, -step * 1.0 / sumFuncs.size());
        VecTools.append(totalGrad, grad);
        VecTools.incscale(totalGrad, gradients[modelId], -1);
        VecTools.assign(gradients[modelId], grad);
      }

      //System.out.println(x);
      System.out.println(sumFuncs.value(x) / step / sumFuncs.dim());
      System.out.flush();
    }

    return x;
  }

  @Override
  public Vec optimize(FuncEnsemble sumFuncs) {
    final Vec x = new ArrayVec(sumFuncs.xdim());
    for (int i = 0; i < x.dim(); i++) {
      x.set(i, rand.nextGaussian());
    }
    return optimizeWithStartX(sumFuncs, x);
  }
}
