package com.expleague.ml.optimization.impl;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.ReguralizerFunc;
import com.expleague.ml.optimization.Optimize;

import java.util.*;
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
  public Vec optimize(final FuncEnsemble<? extends FuncC1> ensemble, ReguralizerFunc reg, final Vec x0) {
    time = System.nanoTime();
    Vec x = VecTools.copy(x0);

    final Vec[] lastGrad = new Vec[ensemble.size()];
    final int[] components = {0};
    final int[] iter = {0};
    final boolean[] occupied = new boolean[ensemble.size()];

    final Vec totalGrad = new ArrayVec(x.dim());
    final Vec L = new ArrayVec(x.dim());
    final List<Vec> lStats = new ArrayList<>();
    VecTools.fill(L, 1./MathTools.EPSILON);
    final ReadWriteLock xLock = new ReentrantReadWriteLock();
    IntStream.range(0, maxIter).parallel().forEach(idx -> {
      Vec grad = new ArrayVec(x.dim());
      Vec step = null;
      final int component;
      xLock.readLock().lock();
      try {
        int next;
        do {
          next = random.nextInt(ensemble.size());
        }
        while (occupied[next]);
        component = next;
        occupied[component] = true;
        ensemble.models[component].gradientTo(x, grad);
        if (lastGrad[component] != null) {
          step = VecTools.copy(grad);
          VecTools.incscale(step, lastGrad[component], -1);
          VecTools.incscale(step, totalGrad, 1. / components[0]);
          double max = VecTools.max(L);
          for (int i = 0; i < step.dim(); i++) {
            double l_i = L.get(i);
            double v = l_i > 0 ? l_i : max;
            step.set(i, step.get(i) / v);
          }
        }
      }
      finally {
        xLock.readLock().unlock();
      }

      Vec gradSparse = VecTools.copySparse(grad);
      xLock.writeLock().lock();
      try {
        final int it = ++iter[0];
        if (step != null) {
          VecTools.incscale(x, step, -this.step * 1/3);
          reg.project(x);
        }
        else components[0]++;

        { // update total
          if (lastGrad[component] != null)
            VecTools.incscale(totalGrad, lastGrad[component], -1);
          VecTools.append(totalGrad, grad);
          lastGrad[component] = gradSparse;
        }

        {
          updateL(L, lStats, grad, gradSparse);
        }

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

  private void updateL(Vec l, List<Vec> lStats, Vec grad, Vec copySparse) {
    for (int i = 0; i < grad.dim(); i++) {
      l.set(i, Math.max(l.get(i), Math.abs(grad.get(i))));
    }
    lStats.add(copySparse);
    if (lStats.size() > 10000) {
      final List<Vec> copy = new ArrayList<>(lStats.subList(lStats.size() - 9000, lStats.size()));
      Vec count = new ArrayVec(grad.dim());
      lStats.clear();
      lStats.addAll(copy);

      VecTools.fill(l, MathTools.EPSILON);
      lStats.forEach(g -> {
        VecIterator nz = g.nonZeroes();
        while (nz.advance()) {
          double abs = Math.abs(nz.value());
          if (abs != 0)
            count.adjust(nz.index(), 1);
          l.set(nz.index(), Math.max(l.get(nz.index()), abs));
        }
      });
      double max = VecTools.max(l);
      for (int i = 0; i < l.dim(); i++) {
        if (count.get(i) < 10) {
          l.set(i, max);
        }
      }
    }
  }

}
