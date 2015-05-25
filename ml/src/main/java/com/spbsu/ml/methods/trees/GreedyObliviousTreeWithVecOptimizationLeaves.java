package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.TransObliviousTree;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: noxoomo
 */

public class GreedyObliviousTreeWithVecOptimizationLeaves<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final GreedyObliviousTree<WeightedLoss<Loss>> base;
  private final FastRandom rand;
  private final VecOptimization<L2> leafLearner;

  public GreedyObliviousTreeWithVecOptimizationLeaves(
    final GreedyObliviousTree<WeightedLoss<Loss>> base,
    final VecOptimization<L2> leafLearner,
    final FastRandom rand) {
    this.base = base;
    this.rand = rand;
    this.leafLearner = leafLearner;
  }

  private final static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Leaves executor", -1);

  @Override
  public TransObliviousTree fit(final VecDataSet ds, final Loss loss) {
    Pair<List<BFOptimizationSubset>, List<BFGrid.BinaryFeature>> tree = base.findBestSubsets(ds, DataTools.bootstrap(loss, rand));
    List<BFGrid.BinaryFeature> conditions = tree.getSecond();
    List<BFOptimizationSubset> subsets = tree.getFirst();
    CountDownLatch latch = new CountDownLatch(subsets.size());
    final Trans[] leafTrans = new Trans[subsets.size()];
    //damn java 7 without unique, filters, etc and autoboxing overhead…
    Set<Integer> uniqueFeatures = new TreeSet<>();
    for (BFGrid.BinaryFeature bf : conditions) {
      uniqueFeatures.add(bf.findex);
    }
    int[] features = new int[uniqueFeatures.size()];
    {
      int j = 0;
      for (Integer i : uniqueFeatures) {
        features[j++] = i;
      }
    }

    //

    for (int i = 0; i < subsets.size(); ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          final BFOptimizationSubset subset = subsets.get(ind);
          int[] points = subset.getPoints();
          if (points.length == 0) {
            leafTrans[ind] = new FakeTrans(points.length, new ArrayVec(1));
            latch.countDown();
          } else {
            Mx subData = subMx(ds.data(), points, features);
            Vec target = loss.target();
            Vec localTarget = subVec(target, points);
            VecDataSetImpl subDataSet = new VecDataSetImpl(subData, ds);
            L2 localLoss = DataTools.newTarget(L2.class, localTarget, subDataSet);
            final Trans result = leafLearner.fit(subDataSet, localLoss);
            leafTrans[ind] = new MappedTrans(result, features);
            latch.countDown();
          }
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
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
    final int[] map;

    public MappedTrans(Trans trans, int[] features) {
      this.trans = trans;
      this.map = features;
    }

    @Override
    public int xdim() {
      return map.length;
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
      return trans.trans(inner);
    }
  }

  class FakeTrans extends Trans.Stub {
    private final int xdim;
    private final Vec result;

    FakeTrans(int xdim, Vec result) {
      this.xdim = xdim;
      this.result = result;
    }

    @Override
    public int xdim() {
      return xdim;
    }

    @Override
    public int ydim() {
      return result.dim();
    }

    @Override
    public Vec trans(Vec x) {
      return copy(result);
    }
  }
}
