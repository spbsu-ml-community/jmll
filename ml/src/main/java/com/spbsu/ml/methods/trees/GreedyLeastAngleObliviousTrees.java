package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
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

public class GreedyLeastAngleObliviousTrees<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  private final GreedyObliviousTree<WeightedLoss<Loss>> base;
  private final FastRandom rand;

  public GreedyLeastAngleObliviousTrees(
    final GreedyObliviousTree<WeightedLoss<Loss>> base,
    final FastRandom rand) {
    this.base = base;
    this.rand = rand;
  }


  private int[] learnPoints(WeightedLoss loss) {
    return loss.points();
  }

  @Override
  public Trans fit(final VecDataSet ds, final Loss loss) {
    final WeightedLoss<Loss> bsLoss = DataTools.bootstrap(loss, rand);
    final Pair<List<BFOptimizationSubset>, List<BFGrid.BinaryFeature>> tree = base.findBestSubsets(ds, bsLoss);
    final List<BFGrid.BinaryFeature> conditions = tree.getSecond();
    final List<BFOptimizationSubset> subsets = tree.getFirst();

    //damn java 7 without unique, filters, etc and autoboxing overhead…
    Set<Integer> uniqueFeatures = new TreeSet<>();
    for (BFGrid.BinaryFeature bf : conditions) {
      if (!bf.row().empty()
        )
        uniqueFeatures.add(bf.findex);
    }
//    //prototype
    while (uniqueFeatures.size() < 6) {
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

    double[] scores = new double[features.length];

    final Subsets[] linearSubsets = new Subsets[subsets.size()];

    final CountDownLatch latch = new CountDownLatch(subsets.size());
    for (int i = 0; i < subsets.size(); ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          linearSubsets[ind] = new Subsets(ds, bsLoss, subsets.get(ind), features);
          latch.countDown();
        }
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }

    for (int i = 0; i < linearSubsets.length; ++i) {
      for (int j = 0; j < scores.length; ++j) {
        scores[j] += linearSubsets[i].scores[j];
      }
    }
    int best = ArrayTools.min(scores);

    Trans[] leavesTrans = new Trans[linearSubsets.length];
    for (int i = 0; i < leavesTrans.length; ++i) {
      leavesTrans[i] = linearSubsets[i].localLinears[best];
    }
    return new TransObliviousTree(conditions, leavesTrans);
  }

  static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("least angle subsets", -1);


}

class Subsets {

  private double multiply(Vec left, Vec right, int[] points) {
    double res = 0;
    for (int i : points) {
      res += left.get(i) * right.get(i);
    }
    return res;
  }

  static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("least angle subsets", -1);

  final BFOptimizationSubset subset;
  final LocalLinear[] localLinears;
  final double[] scores;

  Subsets(final VecDataSet ds,
          final WeightedLoss loss,
          final BFOptimizationSubset subset,
          final int[] features) {
    this.subset = subset;
    localLinears = new LocalLinear[features.length];
    scores = new double[features.length];
    final Vec target = copy(loss.target());
    final Mx data = ds.data();
    final CountDownLatch latch = new CountDownLatch(features.length);
    final int[] points = subset.getPoints();
    final double bias = loss.bestIncrement((WeightedLoss.Stat) subset.total());
    for (int i : points) {
      target.adjust(i, -bias);
    }
    for (int i = 0; i < features.length; ++i) {
      final int ind = i;
      exec.submit(new Runnable() {
        @Override
        public void run() {
          final Vec feature = data.col(features[ind]);
          final double cov = multiply(feature, target, points);
          final double featureNorm2 = multiply(feature, feature, points);
          final double inc = points.length > 50 && featureNorm2 > 0 ? cov / featureNorm2 : 0;
          localLinears[ind] = new LocalLinear(data.columns(), features[ind], inc, bias
          );
          scores[ind] = points.length > 50 && featureNorm2 > 0 ? -(cov * cov / featureNorm2) : 0;
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }
}

class LocalLinear extends Func.Stub {
  final int dim;
  final int condition;
  final double value;
  final double bias;

  public LocalLinear(int dim, int condition, double value, double bias) {
    this.dim = dim;
    this.condition = condition;
    this.value = value;
    this.bias = bias;
  }

  @Override
  public double value(Vec x) {
    return x.get(condition) * value + bias;
  }

  @Override
  public int dim() {
    return dim;
  }
}