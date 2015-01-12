package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.CherryPick;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.CherryRegion;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDCherryRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  private final FastRandom rand = new FastRandom();
  private final double alpha;
  private final double beta;
  private final int maxFailed;

  public GreedyTDCherryRegion(final BFGrid grid) {
    this(grid, 0.02, 0.5, 1);
  }

  public GreedyTDCherryRegion(final BFGrid grid, final double alpha, final double beta, final int maxFailed) {
    this.grid = grid;
    this.alpha = alpha;
    this.beta = beta;
    this.maxFailed = maxFailed;
  }

  public GreedyTDCherryRegion(final BFGrid grid, final double alpha, final double beta) {
    this(grid, alpha, beta, 1);
  }


  @Override
  public CherryRegion fit(final VecDataSet learn, final Loss loss) {
    final List<BitSet> conditions = new ArrayList<>(100);
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    final CherryPick pick = new CherryPick(bds, loss.statsFactory());
    int[] points = ArrayTools.sequence(0, learn.length());

    double currentScore = Double.POSITIVE_INFINITY;
    AdditiveStatistics inside = (AdditiveStatistics) loss.statsFactory().create();
    while (true) {
      final Pair<BitSet, int[]> result = pick.build(new Evaluator<AdditiveStatistics>() {
        @Override
        public double value(final AdditiveStatistics stat) {
          return loss.score(stat);
        }
      }, points, 2);

      final double candidateScore = pick.currentScore;// * (1 + 2.0 / (1 + Math.log(conditions.size() + 1)));
      if (currentScore <= candidateScore + 1e-9) {
        break;
      }
      System.out.println("\n" + result.getFirst());

      points = result.getSecond();
      conditions.add(result.getFirst());
      currentScore = candidateScore;
      inside = pick.inside;
    }


    final CherryRegion region = new CherryRegion(conditions.toArray(new BitSet[conditions.size()]), 1, grid);
//    Vec target = loss.target();
//    double sum = 0;
//    double weight = 0;
//    AdditiveStatistics inside2 = (AdditiveStatistics) loss.statsFactory().create();
//    for (int i = 0; i < bds.original().length(); ++i) {
//      if (region.value(bds, i) == 1) {
//        sum += target.get(i);
//        ++weight;
//      }
//    }
//    System.out.println("\nFound cherry region with " + weight + " points");
//    return new CherryRegion(conditions.toArray(new BitSet[conditions.size()]), weight > 0 ? sum / weight : 0, grid);
    return new CherryRegion(conditions.toArray(new BitSet[conditions.size()]), loss.bestIncrement(inside), grid);
  }


}
