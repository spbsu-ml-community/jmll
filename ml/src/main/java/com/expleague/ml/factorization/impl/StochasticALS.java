package com.expleague.ml.factorization.impl;

import com.expleague.commons.func.impl.WeakListenerHolderImpl;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.Pair;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.factorization.Factorization;

import java.util.function.BiConsumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * Experts League
 * Created by solar on 04.05.17.
 */

public class StochasticALS extends WeakListenerHolderImpl<Pair<Vec, Vec>> implements Factorization {
  private static final Logger log = Logger.getLogger(StochasticALS.class.getName());
  private final FastRandom rng;
  private final double gamma;
  private final int maxIterations;
  private final Cache cache;

  public StochasticALS(FastRandom rng, double gamma, int maxIterations, Cache cache) {
    this.rng = rng;
    this.gamma = gamma;
    this.maxIterations = maxIterations;
    this.cache = cache;
  }

  public StochasticALS(FastRandom rng, double gamma, int maxIterations) {
    this(rng, gamma, maxIterations, null);
  }

  public StochasticALS(FastRandom rng, double gamma, Cache cache) {
    this(rng, gamma, 100000, cache);
  }

  public StochasticALS(FastRandom rng, double gamma) {
    this(rng, gamma, 100000);
  }

  public Pair<Vec, Vec> factorize(final Mx X, final BiConsumer<Integer, Vec> scaledWeakLearner) {
    final int m = X.rows();
    final int n = X.columns();
    if (m < n * 10) {
      log.log(Level.WARNING, "This algorithm is intended to be used for matrices with rows >> columns");
    }

    final Vec v = new ArrayVec(n);
    VecTools.fillGaussian(v, rng);
    VecTools.scale(v, 1. / VecTools.norm(v));
    final double gamma = this.gamma / (2 + X.rows());
    int iteration = 0;

    if (cache != null) {
      Interval.start();
      cache.update(X, scaledWeakLearner);
      Interval.stopAndPrint("Cache update");
    }

    // call cursor update
    X.row(0);

    Interval.start();
    {
      double a;
      double u_hat;

      do {
        iteration++;
        final int i = rng.nextInt(m);
        final double denominator = Math.log(1 + iteration);
        final Vec row = cache != null ? cache.getAnyCached() : X.row(i);
        u_hat = VecTools.multiply(row, v);
        a = 0;
        for (int j = 0; j < n; j++) {
          final double val = gamma * (u_hat * (v.get(j) * u_hat - row.get(j))) / denominator;
          v.adjust(j, -val);
          if (a < Math.abs(val))
            a = Math.abs(val);
        }
        VecTools.scale(v, 1 / VecTools.norm(v)); //TODO
      }
      while (/*a > 0.2 * gamma * u_hat * u_hat && */ iteration < maxIterations);
    }
    Interval.stopAndPrint(iteration + " SALS iterations");

    Interval.start();
    VecTools.scale(v, 1 / VecTools.norm(v));
    final Vec u = new ArrayVec(m);
    IntStream.range(0, m).parallel().forEach(i -> {
      final Vec row = cache != null && cache.containsRow(i) ? cache.getRowFromCache(i) : X.row(i);
      u.set(i, VecTools.multiply(row, v));
    });
    Interval.stopAndPrint("Decomposition calculation");

    return Pair.create(u, v);
  }

  @Override
  public Pair<Vec, Vec> factorize(final Mx X) {
    return factorize(X, null);
  }

  public static class Cache {
    private final int cacheSizeMB;
    private final double updateRate;
    private final FastRandom rng;

    private int cacheSize;
    private Vec[] cachedRows = null;
    private int[] mtcIds = null;
    private int[] ctmIds = null;


    public Cache(final int cacheSizeMB, final double updateRate, final FastRandom rng) {
      this.cacheSizeMB = cacheSizeMB;
      this.updateRate = updateRate;
      this.rng = rng;
    }

    private void initializeCache(final Mx X) {
      cacheSize = Math.min(X.rows(), cacheSizeMB * 1024 * 1024 / (X.columns() * Double.BYTES));
      System.out.println("Initialized cache of size " + cacheSize);

      cachedRows = new Vec[cacheSize];
      mtcIds = new int[X.rows()];
      ctmIds = new int[X.rows()];

      for (int i = 0; i < cacheSize; i++) {
        cachedRows[i] = X.row(i);
        ctmIds[i] = i;
        mtcIds[i] = i;
      }
      for (int i = cacheSize; i < X.rows(); i++) {
        ctmIds[i] = i;
        mtcIds[i] = -1;
      }
    }

    private void swap(final int i, final int j) {
      if (i == j) {
        return;
      }

      if (i < cacheSize && j < cacheSize) {
        final Vec tmp_r = cachedRows[i];
        cachedRows[i] = cachedRows[j];
        cachedRows[j] = tmp_r;
      }

      final int tmp_i = ctmIds[i];
      ctmIds[i] = ctmIds[j];
      ctmIds[j] = tmp_i;

      mtcIds[ctmIds[i]] = i < cacheSize ? i : -1;
      mtcIds[ctmIds[j]] = j < cacheSize ? j : -1;
    }

    private void updateCache(final Mx X, final BiConsumer<Integer, Vec> scaledWeakLearner) {
      final int toUpdate = (int) (updateRate * cacheSize);

      for (int i = cacheSize - toUpdate; i < cacheSize; i++) {
        final int j = rng.nextInt(i + 1);
        swap(i, j);
      }

      IntStream.range(0, cacheSize - toUpdate).parallel().forEach(i -> {
        scaledWeakLearner.accept(ctmIds[i], cachedRows[i]);
      });

      for (int i = cacheSize - toUpdate; i < cacheSize; i++) {
        final int j = i + rng.nextInt(X.rows() - i);
        swap(i, j);
        if (j < cacheSize) {
          scaledWeakLearner.accept(ctmIds[i], cachedRows[i]);
        } else {
          cachedRows[i] = X.row(ctmIds[i]);
        }
      }
    }

    public void update(final Mx X, final BiConsumer<Integer, Vec> scaledWeakLearner) {
      if (cachedRows == null) {
        initializeCache(X);
      } else {
        updateCache(X, scaledWeakLearner);
      }
    }

    public Vec getAnyCached() {
      return cachedRows[rng.nextInt(cacheSize)];
    }

    public boolean containsRow(int i) {
      return mtcIds[i] != -1;
    }

    public Vec getRowFromCache(int i) {
      return cachedRows[mtcIds[i]];
    }

  }
}