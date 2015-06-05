package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.models.ExponentialObliviousTree;
import com.spbsu.ml.models.ObliviousTree;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;

/*Created with IntelliJ IDEA.
    *User:towelenee
    *Date:30.11.13
    *Time:17:48
    *Idea please stop making my code yellow
*/

public class GreedyExponentialObliviousTree extends GreedyObliviousTree<WeightedLoss<? extends L2>> {
  private final double SwapProbability;
  private final ArrayList<ArrayList<Double>> factors;
  private final ArrayList<HashMap<Double, Integer>> upperBoundCache, lowerBoundCache;

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
    upperBoundCache = new ArrayList<>(factors.size());
    lowerBoundCache = new ArrayList<>(factors.size());
    for (int i = 0; i < factors.size(); i++) {
      upperBoundCache.add(new HashMap<Double, Integer>());
      lowerBoundCache.add(new HashMap<Double, Integer>());
    }
  }

  public static <T extends Comparable<T>> int upperBound(final List<T> list, T key) {
    int l = -1, r = list.size();
    while (l != r - 1) {
      final int middle = (l + r) / 2;
      if (list.get(middle).compareTo(key) <= 0) {
        l = middle;
      } else {
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
      } else {
        r = middle;
      }
    }
    return r;
  }

  public void output() throws FileNotFoundException {
    final int factorId = 20;
    final int boarderId = factors.get(factorId).size() / 2;
    final Double boarder = factors.get(factorId).get(boarderId);
    final PrintWriter writer = new PrintWriter("historgram20.tsv");
    double answers[] = new double[factors.get(factorId).size()];
    for (int j = 0; j < factors.get(factorId).size(); j++) {
      answers[j] = getProbabilityOfFit(factorId, factors.get(factorId).get(j), boarder, SwapProbability);
      if (j > 0 && answers[j] > answers[j - 1] + MathTools.EPSILON) {
        getProbabilityOfFit(factorId, factors.get(factorId).get(j), boarder, SwapProbability);
        getProbabilityOfFit(factorId, factors.get(factorId).get(j - 1), boarder, SwapProbability);
        throw new RuntimeException("");
      }
      writer.println(factors.get(factorId).get(j) + "\t" + answers[j]);
      //System.out.println(factors.get(factorId).get(j) + "\t" + getProbabilityOfFit(factorId, factors.get(factorId).get(j), boarder, SwapProbability));
    }
    writer.close();
  }
  private int upperBoundCached(final int factorId, double key) {
    if (!upperBoundCache.get(factorId).containsKey(key)) {
      upperBoundCache.get(factorId).put(key, upperBound(factors.get(factorId), key));
    }
    return upperBoundCache.get(factorId).get(key);
  }

  private int lowerBoundCached(final int factorId, double key) {
    if (!lowerBoundCache.get(factorId).containsKey(key)) {
      lowerBoundCache.get(factorId).put(key, lowerBound(factors.get(factorId), key));
    }
    return lowerBoundCache.get(factorId).get(key);
  }
  public double callMe(int lower_bound, int upper_bound, int t_finish, double p) {
    if (t_finish < lower_bound)
      return sumProgression(p, lower_bound - t_finish, lower_bound);
    else
      return sumProgression(p, 0, lower_bound) + sumProgression(p, 1, t_finish - upper_bound);
  }

  public double getProbability(final int factorId, double x, double target, double swapProbability) {
    int lowerX = lowerBoundCached(factorId, x);
    int upperX = upperBoundCached(factorId, x) - 1;
    int targetIndex = upperBoundCached(factorId, target) - 1;
    if (upperX < lowerX)
      upperX++;
    double normalization = callMe(lowerX, upperX, factors.get(factorId).size() - 1, swapProbability);
    return callMe(lowerX, upperX, targetIndex, swapProbability) / normalization;

  }

  public static double sumProgression(double p, int begin, int end) {
    if (end < begin)
      return 0;
    return (Math.pow(p, begin) - Math.pow(p, end + 1)) / (1 - p);

  }

  private double getProbabilityOfFit(final int factorId, double x, double target, double swapProbability) {
    final double probability = getProbability(factorId, x, target, swapProbability);
    return probability;
  }

  public double[] getProbabilitiesBeingInRegion(final List<BFGrid.BinaryFeature> features, final Vec point) {
    final int numberOfRegions = 1 << features.size();
    double[] probabilities = new double[numberOfRegions];
    Arrays.fill(probabilities, 1.0);
    for (int i = 0; i < features.size(); i++) {
      final int factorId = features.get(i).findex;
      final double condition = features.get(i).condition;
      final double x = point.get(factorId);
      double[] p = new double[2];
      {
        p[0] = getProbabilityOfFit(factorId, x, condition, SwapProbability);
        p[1] = 1 - p[0];
      }
      for (int region = 0; region < 1 << features.size(); region++) {
        probabilities[region] *= p[(region >> (features.size() - i - 1)) & 1];
      }
    }
    return probabilities;
  }

  @Override
  public ObliviousTree fit(final VecDataSet ds, final WeightedLoss<? extends L2> loss) {
    final ObliviousTree tree = super.fit(ds, loss);
    final List<BFGrid.BinaryFeature> features = tree.features();
    final int numberOfRegions = 1 << features.size();

    double[] targetSum = new double[numberOfRegions];
    double[] weightsSum = new double[numberOfRegions];
    for (int i = 0; i < ds.data().rows(); ++i) {
      final double weight = loss.weight(i);
      final double target = loss.target().get(i);
      final Vec point = ds.data().row(i);

      final double[] probabilities = getProbabilitiesBeingInRegion(features, point);
      for (int region = 0; region < numberOfRegions; ++region) {
        final double expWeight = probabilities[region];

        if (expWeight > MathTools.EPSILON) {
          targetSum[region] += target * weight * expWeight;
          weightsSum[region] += weight * expWeight;
        }
      }
    }
    double[] values = new double[numberOfRegions];
    for (int region = 0; region < numberOfRegions; ++region) {
      if (weightsSum[region] > 1e-9) {
        values[region] = targetSum[region] / weightsSum[region];
      }
    }
    return new ExponentialObliviousTree(features, values, tree.based(), this);
    //return new ObliviousTree(features, values, tree.based());
  }
}
