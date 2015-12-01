package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.greedyRegion.BinaryRegion;
import com.spbsu.ml.methods.greedyRegion.RegionBasedOptimization;
import com.spbsu.ml.models.Region;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

public class LassoRegionsForest<Loss extends L2> extends WeakListenerHolderImpl<Trans> implements VecOptimization<Loss> {
  protected final FastRandom rnd;
  private final int count;
  private final RegionBasedOptimization<WeightedLoss<? extends L2>> weak;
  private double lambda;
  private final double alpha;
  private final double tolerance = 1e-5;

  public LassoRegionsForest(RegionBasedOptimization<WeightedLoss<? extends L2>> weak, FastRandom rnd,
                            final int count, final double lambda, final double alpha) {
    this.count = count;
    this.rnd = rnd;
    this.weak = new BinaryRegion<>(weak);
    this.lambda = lambda;
    this.alpha = alpha;
  }

  public LassoRegionsForest(RegionBasedOptimization<WeightedLoss<? extends L2>> weak, FastRandom rnd, final int count) {
    this(weak, rnd, count, 1e-3, 1.0);
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Lasso forest thread", -1);

  @Override
  public Trans fit(final VecDataSet learn, final Loss globalLoss) {
    final Region[] weakModels = new Region[count];
    final Mx transformedData = new VecBasedMx(learn.data().rows(), count);
    final CountDownLatch latch = new CountDownLatch(count);
    for (int i = 0; i < count; ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          weakModels[ind] = weak.fit(learn, DataTools.bootstrap(globalLoss, rnd));
          Mx applied = weakModels[ind].transAll(learn.data());
          for (int row = 0; row < learn.data().rows(); ++row) {
            transformedData.set(row, ind, applied.get(row, 0));
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (Exception e) {
      System.err.println("fit error");
    }
    ElasticNetMethod lasso = new ElasticNetMethod(tolerance, alpha, lambda);
    Vec init = new ArrayVec(count);
    VecTools.fill(init, 0.0);
    Linear model = (Linear) lasso.fit(new VecDataSetImpl(transformedData, learn), globalLoss, init);
    return new Ensemble(weakModels, model.weights);
  }
}