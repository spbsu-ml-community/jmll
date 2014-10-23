package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.Region;
import gnu.trove.list.array.TDoubleArrayList;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  int maxFailed = 1;
  BufferedWriter writer;
  private final FastRandom rand = new FastRandom();
  public GreedyTDRegion(BFGrid grid) {
    try {
      writer = new BufferedWriter(new FileWriter("regionGradients"));
      ;
    } catch (IOException e) {

    }
    this.grid = grid;
  }

  @Override
  public Region fit(VecDataSet learn, final Loss loss) {
    final List<BFGrid.BinaryFeature> conditions = new ArrayList<>(100);
    final List<Boolean> mask = new ArrayList<>();
    double currentScore = Double.POSITIVE_INFINITY;
//    final AdditiveStatistics excluded = (AdditiveStatistics) loss.statsFactory().create();
    BFStochasticOptimizationSubset current;
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    current = new BFStochasticOptimizationSubset(bds, loss, ArrayTools.sequence(0, learn.length()));
    final double[] scores = new double[grid.size()];
    final boolean[] used = new boolean[grid.size()];
    final int[] usedFeatures = new int[grid.rows()];

    int depth = 0;
    while (true) {
      current.visitAllSplits(new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
          if (usedFeatures[bf.findex] != 0) {
//          if s(used[bf.bfIndex] || usedFeatures[bf.findex] > 0) {
            scores[bf.bfIndex] = Double.POSITIVE_INFINITY;
          } else {
            final double leftScore = loss.score(left);
            final double rightScore = loss.score(right);
            scores[bf.bfIndex] = leftScore > rightScore ? rightScore : leftScore;
          }
        }
      });
      final int bestSplit = ArrayTools.min(scores);
      if (bestSplit < 0 || (scores[bestSplit] + 1e-9 >= currentScore && depth > 2))
        break;

      final BFGrid.BinaryFeature bestSplitBF = grid.bf(bestSplit);
      final boolean[] bestSplitMask = new boolean[1];
      current.visitSplit(bestSplitBF, new Aggregate.SplitVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, AdditiveStatistics left, AdditiveStatistics right) {
          final double leftScore = loss.score(left);
          final double rightScore = loss.score(right);
          bestSplitMask[0] = leftScore > rightScore;
        }
      });

      BFStochasticOptimizationSubset outRegion = current.split(bestSplitBF, bestSplitMask[0]);
      if (outRegion == null) {
        break;
      }

//      excluded.append(current.total());
      conditions.add(bestSplitBF);
      usedFeatures[bestSplitBF.findex]++;
//      used[bestSplitBF.bfIndex] = true;
      mask.add(bestSplitMask[0]);
//      current = outRegion;
      currentScore = scores[bestSplit];
      ++depth;
    }


    boolean[] masks = new boolean[conditions.size()];
    for (int i = 0; i < masks.length; i++) {
      masks[i] = mask.get(i);
    }


    Region region = new Region(conditions, masks, 1, 0, -1, currentScore, conditions.size() > 1 ? maxFailed : 0);
    AdditiveStatistics inside = (AdditiveStatistics) loss.statsFactory().create();
    Vec target = inside.getTargets();
    TDoubleArrayList sample = new TDoubleArrayList();
    double sum = 0;
    double weight = 0;

    for (int i = 0; i < bds.original().length(); ++i) {
      if (region.value(bds, i) == 1) {
        double samplWeight = 1.0;//rand.nextPoisson(1.0);
        weight += samplWeight;
        sum += target.get(i) * samplWeight;
//        inside.append(i, 1);
        sample.add(target.get(i));
      }
    }

    try {
      writer.write(Arrays.toString(sample.toArray()) + "\n");
      writer.flush();
    } catch (IOException e) {

    }
//    if (sample.size() < 15) {
    double value = weight > 1 ? sum / weight : loss.bestIncrement(current.total());//loss.bestIncrement(inside);
//    double value = loss.bestIncrement(current.total());
    return new Region(conditions, masks, value, 0, -1, currentScore, conditions.size() > 1 ? maxFailed : 0);
//    } else {
//      sample.sort();
//      double sum = 0;
//      double weight = 0;
//      int left = (int) (Math.ceil(sample.size() * 0.01));
//      int right = (int) (Math.floor(sample.size() * 0.99));
//      while (left > 0 && sample.get(left) == sample.get(left - 1)) --left;
//      while (right < sample.size() && sample.get(right) == sample.get(right - 1)) ++right;
//
//      for (int i = left; i < right; ++i) {
////    for (int i = 0; i < sample.length; ++i) {
//        double entryWeight = 1.0;//rand.nextPoisson(200.0);
////      double entryWeight = rand.nextPoisson(1.0);
//        sum += entryWeight * sample.get(i);
//        weight += entryWeight;
//      }
//      return new Region(conditions, masks, sum / weight, 0, -1, currentScore, conditions.size() > 1 ? maxFailed : 0);
//
//    }

  }

//   double bestInc(Vec target) {
//     double[] sample = new double[points.length];
//     for (int i = 0; i < points.length; ++i) {
//       sample[i] = target.get(points[i]);
//     }
//     double sum = 0;
//     double weight = 0;
//     if (points.length < 4) {
////      return sample[rand.nextInt(sample.length)];
//       for (double elem : sample) {
//         sum += elem;
//       }
//       return sum / points.length;
//     }
////    for (int i = 0; i < points.length; ++i) {
////      sum += sample[i];
////    }
////    sum /= points.length;
////    return sum;
//     Arrays.sort(sample);
//     int left = (int) (Math.ceil(sample.length * 0.01));
//     int right = (int) (Math.ceil(sample.length * 0.99));
//     while (left > 0 && sample[left] == sample[left - 1]) --left;
//     while (right < sample.length && sample[right] == sample[right - 1]) ++right;
//
//     for (int i = left; i < right; ++i) {
////    for (int i = 0; i < sample.length; ++i) {
//       double entryWeight = 1.0;//rand.nextPoisson(200.0);
////      double entryWeight = rand.nextPoisson(1.0);
//       sum += entryWeight * sample[i];
//       weight += entryWeight;
//     }
//     if (weight == 0) {
//       return 0;
//     }
//
//   }
}
