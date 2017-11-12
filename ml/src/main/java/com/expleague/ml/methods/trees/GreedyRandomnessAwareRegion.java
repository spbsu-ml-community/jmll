package com.expleague.ml.methods.trees;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.FeatureBinarization;
import com.expleague.ml.data.BinarizedFeatureDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.distributions.RandomVariable;
import com.expleague.ml.distributions.parametric.DeltaFunction;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.RandomnessAwareVecOptimization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.RandomnessAwareRegion;
import com.expleague.ml.randomnessAware.VecRandomFeatureExtractor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.expleague.commons.math.MathTools.sqr;

/**
 * User: noxoomo
 */
public class GreedyRandomnessAwareRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> implements RandomnessAwareVecOptimization<Loss> {
  private final int maxDepth;
  private final int binarization;
  private final FastRandom random;
  private final List<VecRandomFeatureExtractor> featureExtractors;
  private final boolean sampledAggregate;
  private boolean rebuildStochasticAggregates = false;
  private boolean useBootstrap = false;
  private LeavesType leavesType = LeavesType.DeterministicMean;
  private boolean forceSampledSplit = false;

  public void useBootstrap(final boolean bootstrap) {
    useBootstrap = bootstrap;
  }

  public void forceSampledSplit(final boolean forceSampledSplit) {
    this.forceSampledSplit = forceSampledSplit;
  }

  enum LeavesType {
    BayesianMean,
    DeterministicMean,
    NormalVal
  }

  public GreedyRandomnessAwareRegion(final int depth,
                                     final List<VecRandomFeatureExtractor> featureExtractors,
                                     final int binarization,
                                     final boolean sampledAggregate,
                                     final FastRandom random) {
    this.maxDepth = depth;
    this.featureExtractors = featureExtractors;
    this.binarization = binarization;
    this.sampledAggregate = sampledAggregate;
    this.random = random;
  }

  @Override
  public RandomnessAwareRegion fit(final VecDataSet learn,
                                          final Loss loss) {

    List<RandomnessAwareOptimizationSubset> leaves = new ArrayList<>(1 << maxDepth);
    final List<FeatureBinarization.BinaryFeature> conditions = new ArrayList<>(maxDepth);
    final List<Boolean> masks = new ArrayList<>(maxDepth);

    double currentScore = Double.POSITIVE_INFINITY;

    final BinarizedFeatureDataSet dataSet = buildBinarizedDataSet(learn);
    final BinarizedFeatureDataSet.GridHelper gridHelper = dataSet.gridHelper();

    final double[] scoresLeft = new double[gridHelper.binFeatureCount()];
    final double[] scoresRight = new double[gridHelper.binFeatureCount()];



    RandomnessAwareOptimizationSubset rightSet;
    RandomnessAwareOptimizationSubset leftSet;
    {
      final double weights[] = useBootstrap ? new double[learn.length()] : null;
      if (useBootstrap) {
        for (int i = 0; i < weights.length; ++i) {
          weights[i] = random.nextGamma(1.0);
        }
      }
      final RandomnessAwareOptimizationSubset allPoints = new RandomnessAwareOptimizationSubset(dataSet, loss, ArrayTools.sequence(0, learn.length()), weights, random);

      Arrays.fill(scoresLeft, 0);
      allPoints.visitAllSplits((bf, left, right) -> {
        scoresLeft[gridHelper.binaryFeatureOffset(bf)] += loss.score(left) + loss.score(right);
      });
      final int bestSplit = ArrayTools.min(scoresLeft);
      final FeatureBinarization.BinaryFeature bestSplitBF = gridHelper.binFeature(bestSplit);
      currentScore = scoresLeft[bestSplit];

      leftSet = allPoints;
      rightSet = allPoints.split(bestSplitBF, rebuildStochasticAggregates, forceSampledSplit);
      conditions.add(bestSplitBF);
    }

    for (int level = 1; level < maxDepth; level++) {
      Arrays.fill(scoresLeft, 0.);
      Arrays.fill(scoresRight, 0.);

      final AdditiveStatistics leftTotal = leftSet.total();
      final AdditiveStatistics rightTotal = rightSet.total();

      rightSet.visitAllSplits((bf, left, right) -> {
        scoresRight[gridHelper.binaryFeatureOffset(bf)] += loss.score(left) + loss.score(right) + loss.score(leftTotal);
      });

      leftSet.visitAllSplits((bf, left, right) -> {
        scoresLeft[gridHelper.binaryFeatureOffset(bf)] += loss.score(left) + loss.score(right) + loss.score(rightTotal);
      });



      final int bestSplitLeft = ArrayTools.min(scoresLeft);
      final int bestSplitRight = ArrayTools.min(scoresRight);

      final boolean splitRight = !(scoresLeft[bestSplitLeft] < scoresRight[bestSplitRight]);
      final double bestScore = splitRight ? scoresRight[bestSplitRight] : scoresLeft[bestSplitLeft];
      final int bestSplit = splitRight ? bestSplitRight : bestSplitLeft;

      if (bestSplit < 0 || bestScore + 1e-9 >= currentScore) {
        break;
      }

      final FeatureBinarization.BinaryFeature bestSplitBF = gridHelper.binFeature(bestSplit);
      conditions.add(bestSplitBF);

      if (splitRight) {
        leaves.add(leftSet);
        masks.add(true);
        leftSet = rightSet;
      } else {
        leaves.add(rightSet);
        masks.add(false);
      }
      rightSet = leftSet.split(bestSplitBF, rebuildStochasticAggregates, forceSampledSplit);
      currentScore = bestScore;
    }

    leaves.add(leftSet);
    leaves.add(rightSet);
    masks.add(true);

    final RandomVariable[] step = new RandomVariable[leaves.size()];

    for (int i = 0; i < step.length; i++) {
      final AdditiveStatistics total = leaves.get(i).total();
      final double sum = ((L2.MSEStats) total).sum;
      final double weight = ((L2.MSEStats) total).weight;
      final double bestInc = sum / (weight + 1);
      step[i] = (DeltaFunction) () -> bestInc;
    }
    return new RandomnessAwareRegion(conditions, masks, step);
  }

  private BinarizedFeatureDataSet ds = null;

  private BinarizedFeatureDataSet buildBinarizedDataSet(final VecDataSet learn) {
    if (ds == null) {
      final BinarizedFeatureDataSet.Builder builder = new BinarizedFeatureDataSet.Builder(learn, binarization, random);
      builder.setSampledFlag(sampledAggregate);
      featureExtractors.forEach(builder::addFeature);
      ds = builder.build();
    }
    return ds;
  }

}
