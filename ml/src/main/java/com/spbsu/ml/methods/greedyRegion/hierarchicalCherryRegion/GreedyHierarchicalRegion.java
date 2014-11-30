package com.spbsu.ml.methods.greedyRegion.hierarchicalCherryRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors;
import com.spbsu.ml.models.CherryRegion;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * Created by noxoomo on 24/11/14.
 */
public class GreedyHierarchicalRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  int[][] useCounts;

  public GreedyHierarchicalRegion(BFGrid grid) {
    this.grid = grid;
    useCounts = new int[grid.rows()][];
    for (int f = 0; f < grid.rows(); ++f)
      useCounts[f] = new int[grid.row(f).size() + 1];
  }


  @Override
  public CherryRegion fit(final VecDataSet learn, final Loss loss) {
    final List<RegionLayer> layers = new ArrayList<>(10);
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);


    HierarchicalPick pick = new HierarchicalPick(bds, loss.statsFactory(), 10, 14, useCounts);
    int[] points = ArrayTools.sequence(0, learn.length());

    double information = 0;
    double currentScore = Double.POSITIVE_INFINITY;
    BitSet used = new BitSet();
    while (true) {
      RegionLayer layer = pick.build(new Evaluator<AdditiveStatistics>() {
        @Override
        public double value(AdditiveStatistics stat) {
          if (stat == null)
            return Double.POSITIVE_INFINITY;
          return loss.score(stat);
        }
      }, points);

      if (currentScore <= score(layer.inside) + 1e-9 || layer.information == Double.POSITIVE_INFINITY) {
        break;
      }
      System.out.println("\n" + layer.conditions);
      System.out.println("\n" + weight(layer.inside));
      layers.add(layer);
      currentScore = score(layer.inside);
      information += layer.information;
      int index = 0;
      for (int f = 0; f < grid.rows(); ++f)
        for (int bin = 0; bin <= grid.row(f).size(); ++bin, ++index)
          if (layer.conditions.get(index))
            useCounts[f][bin]++;
      used.or(layer.conditions);
      points = layer.insidePoints;
    }
    BitSet[] conditions = new BitSet[layers.size()];
    for (int i = 0; i < conditions.length; ++i) {
      conditions[i] = layers.get(i).conditions;
    }

//    CherryRegion region = new CherryRegion(conditions, 1, grid);
//    Vec target = loss.target();
//    double sum = 0;
//    double weight = 0;
//    for (int i = 0; i < bds.original().length(); ++i) {
//      if (region.value(bds, i) == 1) {
//        sum += target.get(i);
//        ++weight;
//      }
//    }
    AdditiveStatistics inside = layers.get(layers.size() - 1).inside;
    System.out.println("\nFound cherry region with " + AdditiveStatisticsExtractors.weight(inside) + " bootstrap weight");
    return new CherryRegion(conditions, loss.bestIncrement(inside), grid);
  }

  public double score(AdditiveStatistics stats) {
    double sum = sum(stats);
    double weight = weight(stats);
    return weight > 1 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight)) : 0;
  }
}
