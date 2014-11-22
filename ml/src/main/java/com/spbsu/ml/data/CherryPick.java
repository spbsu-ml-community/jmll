package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import gnu.trove.list.array.TIntArrayList;

import java.util.BitSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * User: solar
 * Date: 26.08.13
 * Time: 22:09
 */
public class CherryPick {
  private final BinarizedDataSet bds;
  private final BFGrid grid;
  private final Factory<AdditiveStatistics> factory;
  //  private final int[] starts;
  private final int[][] binsIndex;
  //  private final int[] inverseIndex;
//  private final byte[] inverseBinIndex;
  public double currentScore = 0;
  final int binsCount;
  public AdditiveStatistics inside;

  public CherryPick(BinarizedDataSet bds, Factory<AdditiveStatistics> factory) {
    this.bds = bds;
    this.grid = bds.grid();
    this.factory = factory;
//    this.starts = new int[grid.rows()];
    this.binsIndex = new int[grid.rows()][];

    int current = 0;
    for (int i = 0; i < grid.rows(); i++) {
      this.binsIndex[i] = new int[grid.row(i).size() + 1];
      for (int bin = 0; bin <= grid.row(i).size(); ++bin) {
        this.binsIndex[i][bin] = current;
        ++current;
      }
    }
    this.binsCount = current;
    inside = factory.create();
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("CherryPick thread", -1);

  @SuppressWarnings("unchecked")
  public <T extends AdditiveStatistics> Pair<BitSet, int[]> build(final Evaluator<T> eval, int[] points, final double lambda) {
    TIntArrayList included = new TIntArrayList();

    int[][] base = new int[grid.rows()][];
    {
      for (int feature = 0; feature < grid.rows(); feature++) {
        base[feature] = new int[grid.row(feature).size() + 1];
        final byte[] bin = bds.bins(feature);
        for (int j = 0; j < points.length; j++) {
          base[feature][bin[points[j]]]++;
        }
      }
    }

    final BitSet conditions = new BitSet(binsCount);
    final T resultStat = (T) factory.create();
    int[] indices2include = points;
    double resultScore = Double.POSITIVE_INFINITY;//eval.value(resultStat);

    do {


      final CountDownLatch latch = new CountDownLatch(binsCount - conditions.cardinality());
      final double[][] scores = new double[grid.rows()][];
      final int[] candidates = indices2include;

      for (int f = 0; f < grid.rows(); ++f) {
        scores[f] = new double[grid.row(f).size() + 1];
        for (int bin = 0; bin <= grid.row(f).size(); ++bin) {
          scores[f][bin] = Double.POSITIVE_INFINITY;
          if (conditions.get(binsIndex[f][bin])) {
            continue;
          }

          conditions.set(binsIndex[f][bin]);
          scores[f][bin] = regularization(conditions, base);
          conditions.clear(binsIndex[f][bin]);
          final int finalBinIndex = bin;
          final int finalFeature = f;

          exec.execute(new Runnable() {
            @Override
            public void run() {
              final byte[] bin = bds.bins(finalFeature);
              final T stat = (T) factory.create().append(resultStat);
              for (int i = 0; i < candidates.length; i++) {
                final int index = candidates[i];
                if (finalBinIndex == bin[index])
                  stat.append(index, 1);
              }
              scores[finalFeature][finalBinIndex] = eval.value(stat) * (1 + lambda * scores[finalFeature][finalBinIndex]);
              latch.countDown();
            }
          });
        }
      }

      try {
        latch.await();
      } catch (InterruptedException e) {
        // skip
      }

      double minScore = Double.POSITIVE_INFINITY;
      int bestFeature = -1;
      int bestBin = -1;
      for (int f = 0; f < grid.rows(); ++f) {
        for (int bin = 0; bin < grid.row(f).size(); ++bin) {
          if (scores[f][bin] < minScore) {
            minScore = scores[f][bin];
            bestFeature = f;
            bestBin = bin;
          }
        }
      }

      if (bestFeature < 0 || minScore >= resultScore)
        break;
      resultScore = minScore;
      conditions.set(binsIndex[bestFeature][bestBin]);
      indices2include = new int[indices2include.length - base[bestFeature][bestBin]];
      int index = 0;
      final byte[] bin = bds.bins(bestFeature);

      for (int i = 0; i < candidates.length; i++) {
        if (bin[candidates[i]] != bestBin) {
          indices2include[index] = candidates[i];
          ++index;
        } else {
          included.add(candidates[i]);
          resultStat.append(candidates[i], 1);
          for (int feature = 0; feature < grid.rows(); feature++) {
            final byte[] bins = bds.bins(feature);
            base[feature][bins[candidates[i]]]--;
          }
        }
      }
    }
    while (true);
    currentScore = resultScore;
    inside = resultStat;
    return new Pair<>(conditions, included.toArray());
  }


  private double regularization(BitSet conditions, int[][] base) {
////    return 0;
//    double result = 0;
//    int index = 0;
//    int count = 0;
//    int realCardinality = 0;
//    for (int i = 0; i < base.length; i++) {
//      final int[] row = base[i];
//      boolean current = conditions.get(index);
//      for (int j = 0; j < row.length; j++, index++) {
//        realCardinality += (current != conditions.get(index) ? 1 : 0);
//        current = conditions.get(index);
//        if (!conditions.get(index))
//          continue;
//        count += row[j];
//        if (j + 1 == row.length || !conditions.get(index + 1)) {
//          result += Math.log(count + 1);
//          count = 0;
//        }
//      }
//    }
//    return result + realCardinality; // information in split + AIC


    double result = 0;
    int index = 0;
    int count = 0;
    int total = 0;
    int realCardinality = 0;
    int inside = 0;
    for (int bin = 0; bin < grid.row(0).size(); ++bin) {
      total += base[0][bin];
    }
//    boolean current = conditions.get(index);
//    for (int f= 0; f < grid.rows();++f)
//      for (int bin =0; bin < grid.row(f).size();++bin) {
//        if (conditions.get(binsIndex[f][bin])) {
//          result += Math.log(base[f][bin]+1);
//          inside += base[f][bin];
//        }
//      }
    int cardinality = 0;
    for (int f = 0; f < grid.rows(); ++f) {
      for (int bin = 0; bin <= grid.row(f).size(); ++bin) {
        if (conditions.get(binsIndex[f][bin])) {
          count += base[f][bin];
          inside += base[f][bin];
          cardinality = 1;
        } else {
          result += Math.log(count + 1);
          realCardinality += cardinality;
          count = 0;
          cardinality = 0;
        }
      }
      result += Math.log(count + 1);
      realCardinality += cardinality;
    }
    return 1.0 / (1 + cardinality) + result;// + Math.log(total - inside + 1);// +  Math.log(excluded+1);// / (1 + realCardinality);
  }
}
