package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ThreadTools;
import com.expleague.commons.math.Trans;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.BFGrid;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.MTA;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.Region;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.IntStream;

public class RegionForest<Loss extends AdditiveLoss> extends VecOptimization.Stub<Loss> {
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
    final Region[] regions = new Region[weakCount];
    final Trans[] weakModels = new Trans[weakCount];
    final CountDownLatch latch = new CountDownLatch(weakCount);
    for (int i = 0; i < weakCount; ++i) {
      final int index = i;
      pool.execute(() -> {
        regions[index] = weak.findRegion(learn, DataTools.bootstrap(globalLoss, rnd));
        latch.countDown();
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    final double[][] samples = new double[weakCount][];
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(weak.grid());
    final L2 base = (L2)(globalLoss instanceof WeightedLoss ? ((WeightedLoss) globalLoss).base() : globalLoss);
    for (int i = 0; i < weakCount; ++i) {
      final Region region = regions[i];
      samples[i] = globalLoss.nzComponents()
          .map(idx -> region.contains(bds, idx) ? idx : -1)
          .filter(idx -> idx >= 0)
          .mapToDouble(idx -> base.target().get(idx))
          .toArray();
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
      weakModels[i] = new Region(Arrays.asList(regions[i].features), regions[i].mask, means[i], 0, 0, 0, regions[i].maxFailed);
    }
    return new Ensemble(weakModels, VecTools.fill(new ArrayVec(weakModels.length), 1.0 / weakCount));
  }

  public Trans fitNaive(final VecDataSet learn, final Loss globalLoss) {
    final Trans[] weakModels = new Trans[weakCount];
    final CountDownLatch latch = new CountDownLatch(weakCount);
    for (int i = 0; i < weakCount; ++i) {
      final int index = i;
      pool.execute(() -> {
        weakModels[index] = weak.fit(learn, DataTools.bootstrap(globalLoss, rnd));
        latch.countDown();
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

