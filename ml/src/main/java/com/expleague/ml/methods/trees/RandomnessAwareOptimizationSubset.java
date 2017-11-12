package com.expleague.ml.methods.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.BinarizedFeatureDataSet;
import com.expleague.ml.data.RandomFeaturesAggregate;
import com.expleague.ml.data.impl.BinarizedFeature;
import com.expleague.ml.data.impl.BinarizedFeatureExpectation;
import com.expleague.ml.data.impl.SampledBinarizedFeature;
import com.expleague.ml.loss.StatBasedLoss;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.Arrays;


/**
 * User: noxoomo
 */
public class RandomnessAwareOptimizationSubset {
  private final BinarizedFeatureDataSet dataSet;
  private final StatBasedLoss<AdditiveStatistics> oracle;
  private final FastRandom random;
  private Subset subset;

  public RandomnessAwareOptimizationSubset(final BinarizedFeatureDataSet dataSet,
                                           final StatBasedLoss<AdditiveStatistics> oracle,
                                           final int[] points,
                                           final double[] weights,
                                           final FastRandom random) {
    this.dataSet = dataSet;
    this.oracle = oracle;
    this.random = random;
    if (weights != null) {
      subset = new Subset(points, weights);
    } else {
      subset = new Subset(points);
    }
  }

  RandomnessAwareOptimizationSubset(final BinarizedFeatureDataSet dataSet,
                                    final StatBasedLoss<AdditiveStatistics> oracle,
                                    final int[] points,
                                    final FastRandom random) {
    this(dataSet, oracle, points, null, random);
  }

  public RandomnessAwareOptimizationSubset(final BinarizedFeatureDataSet dataSet,
                                           final StatBasedLoss<AdditiveStatistics> oracle,
                                           final FastRandom random,
                                           final Subset subset) {
    this.dataSet = dataSet;
    this.oracle = oracle;
    this.random = random;
    this.subset = subset;
  }

  public RandomnessAwareOptimizationSubset split(final FeatureBinarization.BinaryFeature feature,
                                                 boolean rebuildStochasticAggregates,
                                                 boolean forceSampledSplit
                                                 ) {
    final BinarizedFeature binarizedFeature = dataSet.binarizedFeature(feature.owner().owner());
    if (binarizedFeature instanceof BinarizedFeatureExpectation || forceSampledSplit) {
      final SampledBinarizedFeature sampledBinarizedFeature = new SampledBinarizedFeature(feature.owner(), feature.owner().owner().apply(dataSet.owner()));
      return exclusiveSplit(feature, sampledBinarizedFeature, rebuildStochasticAggregates);
//      throw new RuntimeException("Error: soft split is not implemented yet");
    } else {
      return exclusiveSplit(feature, binarizedFeature, rebuildStochasticAggregates);
    }
  }

  private RandomnessAwareOptimizationSubset exclusiveSplit(final FeatureBinarization.BinaryFeature feature,
                                                           final BinarizedFeature binarizedFeature,
                                                           final boolean rebuildStochasticAggregates) {

    final int[] left;
    final int[] right;

    final double[] leftWeights;
    final double[] rightWeights;

    {
      final TIntArrayList leftPoints = new TIntArrayList(subset.points.length);
      final TIntArrayList rightPoints = new TIntArrayList(subset.points.length);

      final TDoubleArrayList leftWeightsList = new TDoubleArrayList(subset.points.length);
      final TDoubleArrayList rightWeightsList = new TDoubleArrayList(subset.points.length);

      if (binarizedFeature instanceof BinarizedFeatureExpectation) {
        throw new RuntimeException("Error: wrong code branch for soft split");
      }

      binarizedFeature.visit(subset.points, new BinarizedFeature.FeatureVisitor() {
        @Override
        public final void accept(final int idx, final int line, final int bin, final double prob) {
          if (feature.value(bin)) {
            rightPoints.add(subset.points[line]);
            rightWeightsList.add(subset.weights[line] * prob);
          }
          else {
            leftPoints.add(subset.points[line]);
            leftWeightsList.add(subset.weights[line] * prob);
          }
        }

        @Override
        public final FastRandom random() {
          return random;
        }
      });
      left = leftPoints.toArray();
      right = rightPoints.toArray();

      leftWeights = leftWeightsList.toArray();
      rightWeights = rightWeightsList.toArray();
    }

    final RandomnessAwareOptimizationSubset rightBro;

    final RandomFeaturesAggregate smallerSubsetAggregate = subset.aggregate;
    if (left.length < right.length) {
      final RandomFeaturesAggregate largeSubsetAggregate = smallerSubsetAggregate.split(left, right, leftWeights, rightWeights, rebuildStochasticAggregates);
      subset = new Subset(left, leftWeights, smallerSubsetAggregate);
      final Subset rightSubset = new Subset(right, rightWeights, largeSubsetAggregate);
      rightBro = new RandomnessAwareOptimizationSubset(dataSet, oracle, random, rightSubset);
    } else {
      final RandomFeaturesAggregate largeSubsetAggregate = smallerSubsetAggregate.split(right, left, rightWeights, leftWeights, rebuildStochasticAggregates);
      subset = new Subset(left, leftWeights, largeSubsetAggregate);
      final Subset rightSubset = new Subset(right, rightWeights, smallerSubsetAggregate);
      rightBro = new RandomnessAwareOptimizationSubset(dataSet, oracle, random, rightSubset);
    }
    return rightBro;
  }



//  private RandomnessAwareOptimizationSubset softSplit(final ) {
//
//    final TIntArrayList left = new TIntArrayList(points.length);
//    final TIntArrayList right = new TIntArrayList(points.length);
////    bds.original().cache().cache(RandomFeaturesBinarizedDataSet);
//    final byte[] bins = bds.bins(feature.findex);
//    for (final int i : points) {
//      if (feature.value(bins[i])) {
//        right.add(i);
//      } else {
//        left.add(i);
//      }
//    }
//    final RandomnessAwareOptimizationSubset rightBro = new RandomnessAwareOptimizationSubset(bds, oracle, right.toArray());
//    aggregate.remove(rightBro.aggregate);
//    points = left.toArray();
//    return rightBro;
//  }


  public void visitAllSplits(final RandomFeaturesAggregate.BinaryFeatureVisitor<? extends AdditiveStatistics> visitor) {
    subset.aggregate.visit(visitor);
  }


  public AdditiveStatistics total() {
    return subset.aggregate.total();
  }

  class Subset {
    private final int[] points;
    private final double[] weights;
    private final RandomFeaturesAggregate aggregate;

    Subset(final int[] points) {
      this.points = points;
      this.weights = new double[points.length];
      Arrays.fill(weights, 1.0);
      this.aggregate = new RandomFeaturesAggregate(oracle.statsFactory(), dataSet, points, weights, random);
    }

    Subset(final int[] points, final double[] weights) {
      this.points = points;
      this.weights = weights;
      this.aggregate = new RandomFeaturesAggregate(oracle.statsFactory(), dataSet, points, weights, random);
    }

    public Subset(final int[] points,
                  final double[] weights,
                  final RandomFeaturesAggregate aggregate) {
      this.points = points;
      this.weights = weights;
      this.aggregate = aggregate;
    }
  }
}
