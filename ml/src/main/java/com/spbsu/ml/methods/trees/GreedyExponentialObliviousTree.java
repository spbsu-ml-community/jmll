package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.models.ObliviousTree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/*Created with IntelliJ IDEA.
    *User:towelenee
    *Date:30.11.13
    *Time:17:48
    *Idea please stop making my code yellow
*/

public class GreedyExponentialObliviousTree extends GreedyObliviousTree<WeightedLoss<? extends L2>>{
  private final double SwapProbability;
  private final ArrayList<ArrayList<Double>> factors;

  public GreedyExponentialObliviousTree(final BFGrid grid, final VecDataSet ds, int depth, double SwapProbability) {
    super(grid, depth);
    this.SwapProbability = SwapProbability;
    factors = new ArrayList<>();
    for (int i = 0; i < ds.data().columns(); i++) {
      final Vec col = ds.data().col(i);
      ArrayList<Double> list = new ArrayList<>(col.dim());
      for (int j = 0; j < col.dim(); j++) {
        list.add(col.get(j));
      }
      Collections.sort(list);
      factors.add(list);
    }
  }

  public static <T extends Comparable<T>> int upperBound(final List<T> list, T key) {
    int l = -1, r = list.size();
    while (l != r - 1) {
      final int middle = (l + r) / 2;
      if (list.get(middle).compareTo(key) <= 0) {
        l = middle;
      }
      else {
        r = middle;
      }
    }
    return r;
  }

  public static <T extends Comparable<T>> int lowerBound(final List<T> list, T key) {
    int l = -1, r = list.size();
    while (l != r - 1) {
      final int middle = (l + r) / 2;
      if (list.get(middle).compareTo(key) < 0) {
        l = middle;
      }
      else {
        r = middle;
      }
    }
    return r;
  }

  public static int findLowerDistance(final List<Double> list, double x, double target) {
    int lowerX = lowerBound(list, x);
    int targetIndex = upperBound(list, target) - 1;
    return Math.abs(lowerX - targetIndex);
  }

  public static int findUpperDistance(final List<Double> list, double x, double target) {
    int upperX = upperBound(list, x);
    int targetIndex = upperBound(list, target) - 1;
    return Math.abs(upperX - targetIndex);
  }

  public static double getProbability(final List<Double> list, double x, double target, double swapProbability) {
    final int lowerDistance = findLowerDistance(list, x, target);
    final int upperDistance = findUpperDistance(list, x, target);
    if (x > target) {
      return sumProgression(swapProbability, lowerDistance, upperDistance);
    } else {
      return sumProgression(swapProbability, upperDistance + 2, lowerDistance + 2);
    }

  }

  public static double sumProgression(double p, int begin, int end) {
    return (Math.pow(p, begin) - Math.pow(p, end)) / (1 - p) / (end - begin);

  }

  public static double getProbabilityOfFit(final List<Double> list, double x, double target, boolean greater, double swapProbability) {
    final double probability = getProbability(list, x, target, swapProbability);
    if ((x > target) == greater) {
      return 1 - probability;
    }
    return probability;
  }

  private double getProbabilityBeingInRegion(final List<BFGrid.BinaryFeature> features, final Vec point, int region) {
    double probability = 1;
    for (int i = 0; i < features.size(); i++) {
      final boolean greater = ((region >> (features.size() - i - 1) & 1) == 1);
      final int factorId = features.get(i).findex;
      final double condition = features.get(i).condition;
      final double x = point.get(factorId);
      probability *= getProbabilityOfFit(factors.get(factorId), x, condition, greater, SwapProbability);
    }
    return probability;
  }

  @Override
  public ObliviousTree fit(final VecDataSet ds, final WeightedLoss<? extends L2> loss) {
    final ObliviousTree tree = super.fit(ds, loss);
    final List<BFGrid.BinaryFeature> features = tree.features();
    final int numberOfRegions = 1 << features.size();

    double[] targetSum = new double[numberOfRegions];
    double[] weightsSum = new double[numberOfRegions];
    for (int i = 0; i < ds.xdim(); ++i) {
      final double weight = loss.weight(i);
      final double target = loss.target().get(i);
      final Vec point = ds.data().row(i);

      for (int region = 0; region < numberOfRegions; ++region) {
        final double expWeight = getProbabilityBeingInRegion(features, point, region);
        if (expWeight < 0 || weight < 0) {
          throw new RuntimeException("");
        }

        targetSum[region] += target * weight * expWeight;
        weightsSum[region] += weight * expWeight;
      }
    }
    double[] values = new double[numberOfRegions];
    for (int region = 0; region < numberOfRegions; ++region) {
      if (weightsSum[region] > 1e-9) {
        values[region] = targetSum[region] / weightsSum[region];
      }
    }

    return new ObliviousTree(features, values, tree.based());
  }
}