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

public class GreedyExponentialObliviousTree extends GreedyObliviousTree<WeightedLoss<L2>> {
  private final double DistCoef;
  private final ArrayList<ArrayList<Double>> factors;

  public GreedyExponentialObliviousTree(final BFGrid grid, final VecDataSet ds, int depth, double DistCoef) {
    super(grid, depth);
    this.DistCoef = DistCoef;
    factors = new ArrayList<>();
    for (int i = 0; i < ds.xdim(); i++) {
      final Vec row = ds.data().row(i);
      ArrayList<Double> list = new ArrayList<>(row.dim());
      for (int j = 0; j < row.dim(); j++) {
        list.set(j, row.get(j));
      }
      Collections.sort(list);
      factors.add(list);
    }
  }

  private <T extends Comparable> int upper_bound(final List<T> list, T key) {
    int i = 0;
    while (i < list.size() && key.compareTo(list.get(i)) <= 0) {
      i++;
    }
    return i;
  }

  private <T extends Comparable> int lower_bound(final List<T> list, T key) {
    int i = 0;
    while (i < list.size() && key.compareTo(list.get(i)) < 0) {
      i++;
    }
    return i;
  }

  double findExpectedDistance(int factorId, double x, double target) {
    int lowerX = lower_bound(factors.get(factorId), x);
    int upperX = upper_bound(factors.get(factorId), x);
    double expectedIndex = (lowerX + upperX) / 2.0;
    int targetIndex = upper_bound(factors.get(factorId), target);

    return Math.abs(expectedIndex - targetIndex);
  }

  double getProbability(int factorId, double x, double target) {
    double distance = findExpectedDistance(factorId, x, target);
    if (x > target) {
      return Math.exp(-DistCoef * (distance));
    } else {
      return Math.exp(-DistCoef * (distance + 1));
    }

  }

  double getProbabilityOfFit(int factorId, double x, double target, boolean greater) {
    final double probability = getProbability(factorId, x, target);
    if ((x > target) == greater) {
      return 1 - probability;
    }
    return probability;
  }

  double getProbabilityBeingInRegion(final List<BFGrid.BinaryFeature> features, final Vec point, int region) {
    double probability = 1;
    for (int i = 0; i < features.size(); i++) {
      final boolean greater = ((region >> i & 1) == 1);
      final int factorId = features.get(i).findex;
      final double condition = features.get(i).condition;
      final double x = point.get(factorId);
      probability *= getProbabilityOfFit(factorId, x, condition, greater);
    }
    return probability;
  }

  @Override
  public ObliviousTree fit(final VecDataSet ds, final WeightedLoss<L2> loss) {
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

        targetSum[region] += target * weight * expWeight;
        weightsSum[region] += weight * expWeight;
      }
    }
    double[] values = new double[numberOfRegions];
    for (int region = 0; region < numberOfRegions; ++region) {
      values[region] = targetSum[region] / weightsSum[region];
    }

    return new ObliviousTree(features, values, tree.based());
  }
}