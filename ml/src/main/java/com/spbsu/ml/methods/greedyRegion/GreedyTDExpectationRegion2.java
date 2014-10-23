package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.BFOptimizationSubset;
import com.spbsu.ml.models.Region;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDExpectationRegion2<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;


  public GreedyTDExpectationRegion2(BFGrid grid) {
    this.grid = grid;
  }

  FastRandom rand = new FastRandom();

  StatBasedLoss bootstrap(Loss loss, double[] prob) {
    int[] poissonWeights = new int[loss.xdim()];
    for (int i = 0; i < loss.xdim(); i++) {
      poissonWeights[i] = rand.nextPoisson(prob[i] * prob.length);
    }
    return new WeightedLoss<>(loss, poissonWeights);
  }


  @Override
  public Region fit(VecDataSet learn, final Loss origLoss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final List<Boolean> mask = new ArrayList<>();
    double currentScore = Double.POSITIVE_INFINITY;
    final AdditiveStatistics excluded = (AdditiveStatistics) origLoss.statsFactory().create();
    BFOptimizationSubset subset;
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    double probs[] = new double[learn.length()];
    Arrays.fill(probs, 1.0 / probs.length);
    StatBasedLoss sampledLoss = bootstrap(origLoss, probs);
    subset = new BFOptimizationSubset(bds, sampledLoss, ArrayTools.sequence(0, learn.length()));

    final double[] scores = new double[grid.size()];
    final boolean[] used = new boolean[grid.size()];
    final short[] usedFeatures = new short[grid.rows()];

    while (true) {
      final StatBasedLoss loss = sampledLoss;
      subset.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
          if (used[bf.bfIndex]) {
            scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
          } else {
            AdditiveStatistics best = loss.score(left) > loss.score(right) ? right : left;
//            scores[bf.bfIndex] = loss.value(excluded)+ Math.min(loss.score(left) + loss.value(right), loss.score(right) + loss.value(left));
            scores[bf.bfIndex] = loss.score(best);
          }
        }
      });
      int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || scores[bestSplit] + 1e-9 >= currentScore)
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean[] bestSplitMask = new boolean[1];
      subset.visitSplit(bestSplitBF, new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
//          bestSplitMask[0] = loss.score(left) + loss.value(right) > loss.score(right) + loss.value(left);
          bestSplitMask[0] = loss.score(left) > loss.score(right);
        }
      });


//      excluded.append(current.total());
      conditions.add(bestSplitBF);
      used[bestSplitBF.bfIndex] = true;
      usedFeatures[bestSplitBF.findex]++;
      mask.add(bestSplitMask[0]);
      subset.reweight(bestSplitBF, bestSplitMask[0], probs);
      sampledLoss = bootstrap(origLoss, probs);
      subset = new BFOptimizationSubset(bds, sampledLoss, ArrayTools.sequence(0, learn.length()));
//      current = outRegion;
      currentScore = scores[bestSplit];
    }
    boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }
    return new Region(conditions, masks, sampledLoss.bestIncrement(subset.total()), 0, -1, currentScore, 1);
//    return new Region(conditions, masks, loss.bestIncrement(current.total()), -1, currentScore);
  }
}
