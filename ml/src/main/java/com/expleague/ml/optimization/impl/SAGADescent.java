package com.expleague.ml.optimization.impl;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.optimization.Optimize;

import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Function;
import java.util.stream.IntStream;

public class SAGADescent implements Optimize<FuncEnsemble<? extends FuncC1>> {
  private final double step;
  private final int maxIter;
  private final Random random;
  private Function<Vec, Vec> projection;
  private long time;

  public SAGADescent(final double step, final int maxIter, final Random random) {
    this.step = step;
    this.maxIter = maxIter;
    this.random = random;
  }

  @Override
  public Vec optimize(final FuncEnsemble<? extends FuncC1> sumFuncs) {
    final Vec x = new ArrayVec(sumFuncs.dim());
    for (int i = 0; i < sumFuncs.dim(); i++) {
      x.set(i, random.nextGaussian());
    }
    return optimize(sumFuncs, x);
  }

  @Override
  public Vec optimize(final FuncEnsemble<? extends FuncC1> ensemble, final Vec x0) {
    time = System.nanoTime();
    Vec xNext = VecTools.copy(x0);
    Vec x = VecTools.copy(x0);

    final Vec[] lastGrad = new Vec[ensemble.size()];
    final int[] components = {0};
    final int[] iter = {0};
//    final double[] totalStep = {0};
    final boolean[] occupied = new boolean[ensemble.size()];

    final Vec totalGrad = new ArrayVec(x.dim());
//    final Vec gradCoordinateInverseFreq = new ArrayVec(x.dim());
    final ReadWriteLock xLock = new ReentrantReadWriteLock();

    IntStream.range(0, maxIter).parallel().forEach(idx -> {
      Vec xCopy;
      Vec grad = new ArrayVec(x.dim());
      Vec step = null;
      final int component;
      xLock.readLock().lock();
      try {
        xCopy = VecTools.copy(x);
        int next;
        do {
          next = random.nextInt(ensemble.size());
        }
        while (occupied[next]);
        component = next;
        occupied[component] = true;
        ensemble.models[component].gradientTo(xCopy, grad);
        if (lastGrad[component] != null) {
          step = VecTools.copy(grad);
          VecTools.incscale(step, lastGrad[component], -1);
          VecTools.incscale(step, totalGrad, 1. / components[0]);
        }
      }
      finally {
        xLock.readLock().unlock();
      }

      xLock.writeLock().lock();
      try {
        final int it = ++iter[0];
        if (step != null) {
          final double scale = this.step / Math.log(1 + it);
          VecTools.incscale(xNext, step, -scale);
          VecTools.assign(x, projection != null ? projection.apply(xNext) : xNext);
          VecTools.incscale(totalGrad, lastGrad[component], -1);
        }
        else components[0]++;
        VecTools.append(totalGrad, grad);
        lastGrad[component] = VecTools.copySparse(grad);
        occupied[component] = false;

        if ((it % 100000) == 0) {
          final long newTime = System.nanoTime();
          System.out.printf("Iteration %d: value=%.6f time=%dms |x|=%.4f\n", it, ensemble.value(x), TimeUnit.NANOSECONDS.toMillis(newTime - time), VecTools.norm(x));
          System.out.println(x);
          time = newTime;
        }
      }
      finally {
        xLock.writeLock().unlock();
      }
    });
    return x;
  }

  @Override
  public void projector(Function<Vec, Vec> projection) {
    this.projection = projection;
  }
}
