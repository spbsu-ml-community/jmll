package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.MultipleVecOptimization;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.TransObliviousTree;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static com.spbsu.commons.math.vectors.VecTools.adjust;

/**
 * User: noxoomo
 */

public class GreedyObliviousTreeWithVecOptimizationLeaves<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final GreedyObliviousTree<WeightedLoss<Loss>> base;
  private final FastRandom rand;
  private final MultipleVecOptimization<L2> leafLearner;

  public GreedyObliviousTreeWithVecOptimizationLeaves(
    final GreedyObliviousTree<WeightedLoss<Loss>> base,
    final MultipleVecOptimization<L2> leafLearner,
    final FastRandom rand) {
    this.base = base;
    this.rand = rand;
    this.leafLearner = leafLearner;
  }

  private final static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Leaves executor", -1);

  @Override
  public TransObliviousTree fit(final VecDataSet ds, final Loss loss) {
    final WeightedLoss<Loss> bsLoss = DataTools.bootstrap(loss, rand);
    final Pair<List<BFOptimizationSubset>, List<BFGrid.BinaryFeature>> tree = base.findBestSubsets(ds, bsLoss);
    final List<BFGrid.BinaryFeature> conditions = tree.getSecond();
    final List<BFOptimizationSubset> subsets = tree.getFirst();
    final CountDownLatch latch = new CountDownLatch(subsets.size());
    final Trans[] leafTrans = new Trans[subsets.size()];
    //damn java 7 without unique, filters, etc and autoboxing overhead…
    Set<Integer> uniqueFeatures = new TreeSet<>();
    for (BFGrid.BinaryFeature bf : conditions) {
      if (!bf.row().empty())
        uniqueFeatures.add(bf.findex);
    }
//    //prototype
    while (uniqueFeatures.size() < 10) {
      final int feature = rand.nextInt(ds.data().columns());
      if (!base.grid.row(feature).empty())
        uniqueFeatures.add(feature);
    }

    final int[] features = new int[uniqueFeatures.size()];
    {
      int j = 0;
      for (Integer i : uniqueFeatures) {
        features[j++] = i;
      }
    }
    {
      final VecDataSet[] datas = new VecDataSet[subsets.size()];
      final L2[] losses = new L2[subsets.size()];

      for (int i = 0; i < subsets.size(); ++i) {
        final int ind = i;
        exec.submit(new Runnable() {
          @Override
          public void run() {
            final BFOptimizationSubset subset = subsets.get(ind);
            int[] points = subset.getPoints();
            Mx subData = subMx(ds.data(), points, features);
            Vec target = loss.target();
            Vec localTarget = subVec(target, points);
            final double bias = bsLoss.bestIncrement((WeightedLoss.Stat) subset.total());
            adjust(localTarget, -bias);
            VecDataSetImpl subDataSet = new VecDataSetImpl(subData, ds);
            L2 localLoss = DataTools.newTarget(L2.class, localTarget, subDataSet);
            datas[ind] = subDataSet;
            losses[ind] = localLoss;
            latch.countDown();
          }
        });
      }

      try {
        latch.await();
      } catch (InterruptedException e) {
        e.printStackTrace();
      }

      Trans[] result = leafLearner.fit(datas, losses);

      for (int i = 0; i < subsets.size(); ++i) {
        leafTrans[i] = new MappedTrans(result[i], features, bsLoss.bestIncrement((WeightedLoss.Stat) subsets.get(i).total()), ds.xdim());
      }
    }

    return new TransObliviousTree(conditions, leafTrans);
  }

  private Vec subVec(Vec target, int[] points) {
    Vec result = new ArrayVec(points.length);
    for (int i = 0; i < points.length; ++i) {
      result.set(i, target.get(points[i]));
    }
    return result;
  }

  private Mx subMx(Mx base, int[] points, int[] features) {
    Mx result = new VecBasedMx(points.length, features.length);
    for (int i = 0; i < points.length; ++i) {
      for (int j = 0; j < features.length; ++j) {
        result.set(i, j, base.get(points[i], features[j]));
      }
    }
    return result;
  }

  private class MappedTrans extends Trans.Stub {
    final Trans trans;
    final double bias;
    final int[] map;
    final int xdim;

    public MappedTrans(Trans trans, int[] features, double bias, int xdim) {
      this.trans = trans;
      this.map = features;
      this.bias = bias;
      this.xdim = xdim;
    }

    @Override
    public int xdim() {
      return xdim;
    }

    @Override
    public int ydim() {
      return trans.ydim();
    }

    @Override
    public Vec trans(Vec x) {
      Vec inner = new ArrayVec(map.length);
      for (int i = 0; i < map.length; ++i) {
        inner.set(i, x.get(map[i]));
      }
      return adjust(trans.trans(inner), bias);
    }
  }
}

