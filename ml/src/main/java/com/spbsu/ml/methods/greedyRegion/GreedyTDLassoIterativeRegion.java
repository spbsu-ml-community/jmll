package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import com.spbsu.ml.models.Region;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * Created by noxoomo on 10/02/15.
 */

public class GreedyTDLassoIterativeRegion<Loss extends StatBasedLoss> extends RegionBasedOptimization<Loss> {
  protected final BFGrid grid;
  private int maxDepth= 100;
  private int maxFailed = 1 ;
  private double lambda = 1e-3;

  public GreedyTDLassoIterativeRegion(final BFGrid grid) {
    this(grid, 1,100);
  }

  public GreedyTDLassoIterativeRegion(final BFGrid grid,  final int maxFailed, final int maxDepth) {
    this.grid = grid;
    this.maxDepth = maxDepth;
    this.maxFailed = maxFailed;
  }


  @Override
  public Region fit(final VecDataSet learn, final Loss loss) {
    Region current = new Region(new ArrayList<BFGrid.BinaryFeature>(), null, 0, 0, 0, Double.POSITIVE_INFINITY, -1);
    for (int failed = 0; failed < maxFailed ;++failed) {
      Region next = fitWeak(learn, loss, current, current.maxFailed + 1);
      if (next.score + 1e-9f >= current.score) {
        return current;
      }
      current = next;
    }
    return current;
  }


  public Region fitWeak(final VecDataSet learn, final Loss loss, final Region init, final int maxFailed) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final boolean[] usedBF = new boolean[grid.size()];
    final List<Boolean> mask = new ArrayList<>();
    for (int i = 0; i < init.features.length; ++i) {
      conditions.add(init.features[i]);
      usedBF[init.features[i].bfIndex] = true;
      mask.add(init.mask[i]);
    }
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    final BFWeakConditionsOptimizationRegion current =
            new BFWeakConditionsOptimizationRegion(bds, loss, ArrayTools.sequence(0, learn.length()), init.features, init.mask, maxFailed);
    AdditiveStatistics currentInside = (AdditiveStatistics) loss.statsFactory().create();
    AdditiveStatistics currentCritical = (AdditiveStatistics) loss.statsFactory().create();
    AdditiveStatistics currentOutside = (AdditiveStatistics) loss.statsFactory().create();
    currentInside.append(current.total());
    currentOutside.append(current.excluded);
    currentCritical.append(currentInside);
    currentCritical.remove(current.nonCriticalTotal);
    final boolean[] isRight = new boolean[grid.size()];
    final double[] scores = new double[grid.size()];
    double currentScore = lassoScore(currentInside,currentOutside,calcLambda(currentInside,currentOutside, conditions.size(),maxFailed));//loss.score(currentInside) * (1 +  Math.log(weight(currentInside) + 1));// + (conditions.size() + maxFailed) * Math.log(alpha));
    while (conditions.size() < maxFailed + maxDepth) {
      current.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(final BFGrid.BinaryFeature bf, final AdditiveStatistics left, final AdditiveStatistics right) {
          if (usedBF[bf.bfIndex]) {
            scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
          } else {
            final double leftScore;
            {
              final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
              in.append(current.nonCriticalTotal);
              in.append(left);
              final AdditiveStatistics out = (AdditiveStatistics) loss.statsFactory().create();
              out.append(current.excluded);
              out.append(right);
              leftScore = lassoScore(in,out,calcLambda(in,out,conditions.size()+1,maxFailed));
            }
            final double rightScore;
            {
              final AdditiveStatistics in = (AdditiveStatistics) loss.statsFactory().create();
              in.append(current.nonCriticalTotal);
              in.append(right);
              final AdditiveStatistics out = (AdditiveStatistics) loss.statsFactory().create();
              out.append(current.excluded);
              out.append(left);
              rightScore = lassoScore(in,out,calcLambda(in,out,conditions.size()+1,maxFailed));
            }
            scores[bf.bfIndex] = leftScore > rightScore ? rightScore : leftScore;
            isRight[bf.bfIndex] = leftScore > rightScore;
          }
        }
      });

      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0)
        break;


      if ((scores[bestSplit] + 1e-9 >= currentScore))
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean bestSplitMask = isRight[bestSplitBF.bfIndex];

      final BFOptimizationSubset outRegion = current.split(bestSplitBF, bestSplitMask);
      if (outRegion == null) {
        break;
      }

      conditions.add(bestSplitBF);
      usedBF[bestSplitBF.bfIndex] = true;
      mask.add(bestSplitMask);
      currentScore = scores[bestSplit];
      currentInside = (AdditiveStatistics) loss.statsFactory().create();
      currentInside.append(current.total());
      currentOutside = (AdditiveStatistics) loss.statsFactory().create();
      currentOutside.append(current.excluded);
    }
    final boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }
    final Region region = new Region(conditions, masks,
            lassoIncrement(currentInside, currentOutside, calcLambda(currentInside,currentOutside,conditions.size(),maxFailed)), 0, -1, currentScore, maxFailed);
    return region;
  }


  double calcLambda(AdditiveStatistics inside,AdditiveStatistics outside, int conditions, int maxFailed) {
    return lambda;
  }

  double lassoIncrement(AdditiveStatistics inside,AdditiveStatistics outside, double lambda) {
    final double s = sum(inside);
    final double w = weight(inside);
    final double total = w+weight(outside);
    lambda *= (Math.log(total+1) - Math.log(w+1));
    final double beta = w > 0 ? softThreshold(s,lambda*(total)) / w : 0;
    return beta;
  }
  double lassoScore(AdditiveStatistics inside,AdditiveStatistics outside, double lambda) {
    final double s = sum(inside);
    final double w = weight(inside);
    final double total = w+weight(outside);
    lambda *= (Math.log(total+1) - Math.log(w+1));
    final double beta = w > 0 ? softThreshold(s,lambda*(total)) / w : 0;
    final double score = (-2 * beta * s + beta * beta * w)  ;// + lambda * Math.abs(beta);
    return score;
  }

  private double softThreshold(final double z, final double j) {
    final double sgn = Math.signum(z);
    return sgn * Math.max(sgn * z - j, 0);
  }


}
