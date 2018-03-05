package com.expleague.ml.optimization.impl;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.optimization.Optimize;

import java.util.Random;
import java.util.stream.IntStream;

public class FullGradientDescent implements Optimize<FuncEnsemble<? extends FuncC1>> {
  private final Random random;
  private final double step ;
  private final int epochCount;

  public FullGradientDescent(final Random random, final double step, final int epochCount) {
    this.random = random;
    this.step = step;
    this.epochCount = epochCount;
  }

  @Override
  public Vec optimize(FuncEnsemble<? extends FuncC1> sumFuncs) {
    final Vec x = new ArrayVec(sumFuncs.dim());
    for (int i = 0; i < sumFuncs.dim(); i++) {
      x.set(i, random.nextGaussian());
    }
    return optimize(sumFuncs, x);
  }

  @Override
  public Vec optimize(FuncEnsemble<? extends FuncC1> sumFuncs, Vec x0) {
    final Vec x = VecTools.copy(x0);
    double curLoss = getLoss(sumFuncs, x);
    for (int epoch = 0; epoch < epochCount; epoch++) {
      final Vec grad = IntStream.range(0, sumFuncs.size())
          .parallel()
          .mapToObj(i -> sumFuncs.models[i].gradient(x))
          .reduce(VecTools::append)
          .get();
      VecTools.incscale(x, grad, -step / sumFuncs.size());
      final double newLoss = getLoss(sumFuncs, x);
      System.out.printf(
          "Full gradient descent epoch %d oldLoss %.6f newLoss %.6f\n", epoch, curLoss, newLoss
      );
      if (curLoss > newLoss) {
        curLoss = newLoss;
      } else {
        VecTools.incscale(x, grad, step / sumFuncs.size());
        return x;
      }
    }

    return x;
  }

  private double getLoss(FuncEnsemble<? extends FuncC1> sumFuncs, Vec x) {
    return IntStream.range(0, sumFuncs.size())
        .parallel()
        .mapToDouble(i -> sumFuncs.models[i].value(x))
        .sum() / sumFuncs.size();
  }
}
