package com.expleague.ml.methods.greedyRegion;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.util.ArrayTools;
import com.expleague.ml.BFGrid;
import com.expleague.ml.Binarize;
import com.expleague.ml.data.Aggregate;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.AdditiveLoss;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.CNF;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDCherryRegion<Loss extends AdditiveLoss<?>> extends VecOptimization.Stub<Loss> {
  public final BFGrid grid;
  private int lambda = 1;

  public GreedyTDCherryRegion(final BFGrid grid) {
    this.grid = grid;
  }

  private int[] learnPoints(Loss loss, VecDataSet ds) {
    if (loss instanceof WeightedLoss) {
      return ((WeightedLoss) loss).points();
    } else return ArrayTools.sequence(0, ds.length());
  }

  @Override
  public CNF fit(final VecDataSet learn, final Loss loss) {
    final List<CNF.Clause> conditions = new ArrayList<>(100);
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    final BFGrid grid = bds.grid();

    double currentScore = Double.POSITIVE_INFINITY;
    double complexity = 0;
    double bestIncrement = 0;
    final TIntArrayList activeSet = new TIntArrayList(IntStream.range(0, loss.components()).toArray());
    final TIntHashSet usedFeatures = new TIntHashSet();

    while (true) { // conjunction iterations
      final Aggregate agg = new Aggregate(bds, loss.statsFactory());
      agg.append(activeSet.toArray());

      CNF.Condition bestCondition = null;
      double bestConditionScore = currentScore;
      for (int findex = 0; findex < grid.rows(); ++findex) { // for each feature
        final BFGrid.Row feature = grid.row(findex);
        final BitSet usedBins = new BitSet(feature.size());
        final CNF.Condition currentFeatureBestCondition = new CNF.Condition(grid.row(findex), usedBins);
        final AdditiveStatistics inside = loss.statsFactory().apply(findex);
        final int newFeatureSetSize = usedFeatures.contains(findex) ? usedFeatures.size() : usedFeatures.size() + 1;
        if (newFeatureSetSize > 7) // don't allow combination of too many features, this needs to be reworked
          continue;

        int bestBin;
        double bestFeatureScore = currentScore;

        do { // greedy looking for the best disjunction for feature
          bestBin = -1;
          for (int bin = 0; bin < feature.size(); bin++) {
            if (usedBins.get(bin)) // already in the optimal set
              continue;
            inside.append(agg.bin(findex, bin));
            //noinspection unchecked
            double score = ((AdditiveLoss<AdditiveStatistics>) loss).score(inside);
            usedBins.set(bin);
            score *= (1 + lambda * regularization(currentFeatureBestCondition, complexity));
            if (score < bestFeatureScore) {
              bestFeatureScore = score;
              bestBin = bin;
            }
            usedBins.clear(bin);
            inside.remove(agg.bin(findex, bin));
          }
          if (bestBin > 0)
            usedBins.set(bestBin);
        }
        while(bestBin >= 0);
        if (bestConditionScore > bestFeatureScore) {
          bestCondition = currentFeatureBestCondition;
          bestConditionScore = bestFeatureScore;
          //noinspection unchecked
          bestIncrement = ((AdditiveLoss<AdditiveStatistics>) loss).bestIncrement(inside);
        }
      }

      if (bestCondition == null)
        break;
      complexity += bestCondition.cardinality();
      complexity++;
      System.out.println("\nAdded clause " + bestCondition);
      conditions.add(new CNF.Clause(grid, bestCondition));
    }
    return new CNF(conditions.toArray(new CNF.Clause[conditions.size()]), bestIncrement, 0, grid);
  }

  private double regularization(CNF.Condition currentFeatureBestCondition, double totalComplexity) {
    return Math.log(1. + currentFeatureBestCondition.cardinality() + 1 + totalComplexity);
  }
}