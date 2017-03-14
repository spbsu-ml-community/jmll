package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.linearRegressionExperiments.MultipleValidationRidgeRegression;
import com.spbsu.ml.models.TransObliviousTree;
import gnu.trove.list.array.TIntArrayList;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static com.spbsu.commons.math.vectors.VecTools.adjust;

/**
 * User: noxoomo
 */

public class GreedyObliviousTreeValidationRidgeLeaves<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final GreedyObliviousTree<WeightedLoss<Loss>> base;
  private final FastRandom rand;

  public GreedyObliviousTreeValidationRidgeLeaves(
    final GreedyObliviousTree<WeightedLoss<Loss>> base,
    final FastRandom rand) {
    this.base = base;
    this.rand = rand;
  }

  private final static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Leaves executor", -1);

  private int[] oobPoints(WeightedLoss<Loss> loss) {
    final TIntArrayList result = new TIntArrayList(loss.dim() + 1000);
    for (int i = 0; i < loss.dim(); i++) {
      if (loss.weight(i) == 0)
        result.add(i);
    }
    return result.toArray();
  }

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
      if (bf.row().size() > 2)
        uniqueFeatures.add(bf.findex);
    }
//    //prototype
    if (ds.data().rows() > 20) {
      while (uniqueFeatures.size() < 6) {
        int addFeature = rand.nextInt(ds.data().columns());
        if (base.grid.row(addFeature).size() > 2) {
          uniqueFeatures.add(addFeature);
        }
      }
    }

    final int[] features = new int[uniqueFeatures.size()];
    {
      int j = 0;
      for (Integer i : uniqueFeatures) {
        features[j++] = i;
      }
    }

    final List<BFOptimizationSubset> oobSubsets;
    final int[] oobPoints = oobPoints(bsLoss);

    {
      final BinarizedDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarize(base.grid);
      List<BFOptimizationSubset> leaves = new ArrayList<>(1);
      leaves.add(new BFOptimizationSubset(bds, loss, oobPoints));

      for (int i = 0; i < conditions.size(); ++i) {
        final List<BFOptimizationSubset> next = new ArrayList<>(leaves.size() * 2);
        final ListIterator<BFOptimizationSubset> iter = leaves.listIterator();
        while (iter.hasNext()) {
          final BFOptimizationSubset subset = iter.next();
          next.add(subset);
          next.add(subset.split(conditions.get(i)));
        }
        leaves = next;
      }
      oobSubsets = leaves;
    }

    {
      final VecDataSet[] datas = new VecDataSet[subsets.size()];
      final VecDataSet[] valDatas = new VecDataSet[subsets.size()];
      final L2[] losses = new L2[subsets.size()];
      final L2[] valLosses = new L2[subsets.size()];


      for (int i = 0; i < subsets.size(); ++i) {
        final int ind = i;
        exec.submit(new Runnable() {
          @Override
          public void run() {
            {
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

              final BFOptimizationSubset valSubset = oobSubsets.get(ind);
              int[] valPoints = valSubset.getPoints();
              Mx valData = subMx(ds.data(), valPoints, features);
              Vec valTarget = subVec(target, valPoints);
              adjust(valTarget, -bias);
              valDatas[ind] = new VecDataSetImpl(valData, ds);
              valLosses[ind] = DataTools.newTarget(L2.class, valTarget, valDatas[ind]);
            }
            latch.countDown();
          }
        });
      }

      try {
        latch.await();
      } catch (InterruptedException e) {
        e.printStackTrace();
      }

      MultipleValidationRidgeRegression ridgeRegression = new MultipleValidationRidgeRegression();
      Trans[] result = ridgeRegression.fit(datas, losses, valDatas, valLosses);

      for (int i = 0; i < subsets.size(); ++i) {
        leafTrans[i] = new MappedTrans(result[i], features, bsLoss.bestIncrement((WeightedLoss.Stat) subsets.get(i).total()));
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

    public MappedTrans(Trans trans, int[] features, double bias) {
      this.trans = trans;
      this.map = features;
      this.bias = bias;
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
      return adjust(trans.trans(inner), bias);
    }
  }

}
