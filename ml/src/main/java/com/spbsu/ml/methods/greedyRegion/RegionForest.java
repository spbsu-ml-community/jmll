package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.MTA;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.Region;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;


public class RegionForest<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  public enum MeanMethod {
    MTAConst, Stein, Naive, MTAMinMax
  }

  protected final FastRandom rnd;
  private final GreedyTDRegion<WeightedLoss> weak;
  private final int weakCount;
  private final ThreadPoolExecutor pool;
  private final MeanMethod meanMethod;


  public RegionForest(final BFGrid grid, final FastRandom rnd, final int weakCount) {
    this(grid, rnd, weakCount, MeanMethod.Naive);
  }

  public RegionForest(final BFGrid grid, final FastRandom rnd, final int weakCount, final double alpha, final double beta) {
    this(grid, rnd, weakCount, MeanMethod.Naive, alpha, beta, 1);
  }

  public RegionForest(final BFGrid grid, final FastRandom rnd, final int weakCount, final MeanMethod meanMethod) {
    this(grid, rnd, weakCount, meanMethod, 0.02, 0.05, 1);
  }

  public RegionForest(final BFGrid grid, final FastRandom rnd, final int weakCount, final MeanMethod meanMethod, final double alpha, final double beta, final int maxFailed) {
    this.weak = new GreedyTDRegion<>(grid, alpha, beta, maxFailed);
    this.weakCount = weakCount;
    this.rnd = rnd;
    pool = ThreadTools.createBGExecutor("RF pool", weakCount);
    this.meanMethod = meanMethod;
  }

  public RegionForest(final BFGrid grid, final FastRandom rnd, final int weakCount, final double alpha, final double beta, final int maxFailed) {
    this(grid, rnd, weakCount, MeanMethod.Naive, alpha, beta, maxFailed);
  }


  @Override
  public Trans fit(final VecDataSet learn, final Loss globalLoss) {
    switch (meanMethod) {
      case Naive: {
        return fitNaive(learn, globalLoss);
      }
      default: {
        return fitMTA(learn, globalLoss);
      }
    }

  }

  public Trans fitMTA(final VecDataSet learn, final Loss globalLoss) {
    final GreedyTDRegion.RegionStats[] regions = new GreedyTDRegion.RegionStats[weakCount];
    final Trans[] weakModels = new Trans[weakCount];
    final CountDownLatch latch = new CountDownLatch(weakCount);
    for (int i = 0; i < weakCount; ++i) {
      final int index = i;
      pool.execute(new Runnable() {
        @Override
        public void run() {
          regions[index] = weak.findRegion(learn, DataTools.bootstrap(globalLoss, rnd));
          latch.countDown();
        }
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    final double[][] samples = new double[weakCount][];
    for (int i = 0; i < weakCount; ++i) {
      samples[i] = regions[i].inside.toArray();
    }
    final MTA mta = new MTA(samples);
    final double[] means;
    switch (meanMethod) {
      case Stein: {
        means = mta.stein();
        break;
      }
      case MTAConst: {
        means = mta.mtaConst();
        break;
      }
      case MTAMinMax: {
        means = mta.mtaMiniMax();
        break;
      }
      default: {
        means = mta.classic();
        break;
      }
    }
    for (int i = 0; i < weakCount; ++i) {
      weakModels[i] = new Region(regions[i].conditions, regions[i].mask, means[i], 0, 0, 0, regions[i].maxFailed);
    }
    return new Ensemble(weakModels, VecTools.fill(new ArrayVec(weakModels.length), 1.0 / weakCount));
  }

  public Trans fitNaive(final VecDataSet learn, final Loss globalLoss) {
    final Trans[] weakModels = new Trans[weakCount];
    final CountDownLatch latch = new CountDownLatch(weakCount);
    for (int i = 0; i < weakCount; ++i) {
      final int index = i;
      pool.execute(new Runnable() {
        @Override
        public void run() {
          weakModels[index] = weak.fit(learn, DataTools.bootstrap(globalLoss, rnd));
          latch.countDown();
        }
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    return new Ensemble(weakModels, VecTools.fill(new ArrayVec(weakModels.length), 1.0 / weakCount));
  }
}

