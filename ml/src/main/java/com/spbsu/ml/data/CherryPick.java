package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ArrayTools;
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
  private final int binsCount;
  private final int[] starts;
  private final int[] inverseIndex;
  private final byte[] inverseBinIndex;
  public double currentScore = 0;
  public AdditiveStatistics inside;

  public CherryPick(BinarizedDataSet bds, Factory<AdditiveStatistics> factory) {
    this.bds = bds;
    this.grid = bds.grid();
    this.factory = factory;
    this.starts = new int[grid.rows()];
    int tmp = 0;
    for (int i = 0; i < grid.rows(); i++) {
      starts[i] = tmp;
      tmp += grid.row(i).size() + 1;
    }
    binsCount = tmp;
    inverseIndex = new int[binsCount];
    inverseBinIndex = new byte[binsCount];
    for (int i = 0; i < grid.rows(); ++i) {
      for (int bin = 0; bin <= grid.row(i).size(); ++bin) {
        inverseIndex[starts[i] + bin] = i;
        inverseBinIndex[starts[i] + bin] = (byte) bin;
      }
    }
    inside = factory.create();
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("CherryPick thread", -1);

  @SuppressWarnings("unchecked")
  public <T extends AdditiveStatistics> Pair<BitSet, int[]> build(final Evaluator<T> eval, int[] points, final double lambda) {
    TIntArrayList included = new TIntArrayList();

    int[] base = new int[binsCount];
    {
      for (int feature = 0; feature < grid.rows(); feature++) {
        final byte[] bin = bds.bins(feature);
        for (int j = 0; j < points.length; j++) {
          base[starts[feature] + bin[points[j]]]++;
        }
      }
    }

    final BitSet conditions = new BitSet(binsCount);
    final T resultStat = (T) factory.create();
    int[] indices2include = points;
    double resultScore = Double.POSITIVE_INFINITY;//eval.value(resultStat);

    do {


      final CountDownLatch latch = new CountDownLatch(binsCount - conditions.cardinality());
      final double[] scores = new double[binsCount];
      final int[] candidates = indices2include;

      for (int f = 0; f < grid.rows(); ++f) {
        for (int bin = 0; bin <= grid.row(f).size(); ++bin) {
          scores[starts[f] + bin] = Double.POSITIVE_INFINITY;
          if (conditions.get(starts[f] + bin)) {
            continue;
          }

          conditions.set(starts[f] + bin);
          scores[starts[f] + bin] = lambda * regularization(conditions, base);
          conditions.clear(starts[f] + bin);
          final int finalBinIndex = bin;
          final int feature = f;

          exec.execute(new Runnable() {
            @Override
            public void run() {
              final byte[] bin = bds.bins(feature);
              final T stat = (T) factory.create().append(resultStat);
              for (int i = 0; i < candidates.length; i++) {
                final int index = candidates[i];
                if (finalBinIndex == bin[index])
                  stat.append(index, 1);
              }
              scores[starts[feature] + finalBinIndex] = eval.value(stat) * (1 + lambda * scores[starts[feature] + finalBinIndex]);
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

      final int min = ArrayTools.min(scores);
      if (min < 0 || scores[min] >= resultScore)
        break;
      resultScore = scores[min];
      conditions.set(min);
      indices2include = new int[indices2include.length - base[min]];
      int index = 0;
      final byte[] bin = bds.bins(inverseIndex[min]);

      for (int i = 0; i < candidates.length; i++) {
        if (bin[candidates[i]] != inverseBinIndex[min]) {
          indices2include[index] = candidates[i];
          ++index;
        } else {
          included.add(candidates[i]);
          resultStat.append(candidates[i], 1);
          for (int feature = 0; feature < grid.rows(); feature++) {
            final byte[] bins = bds.bins(feature);
            base[starts[feature] + bins[candidates[i]]]--;
          }
        }
      }
    }
    while (true);
    currentScore = resultScore;
    inside = resultStat;
    return new Pair<>(conditions, included.toArray());
  }


  private double regularization(BitSet conditions, int[] base) {
//    double result = 0;
//    int index = 0;
//    int count = 0;
//    int excluded = 0;
//    int total = 0;
//    int realCardinality = 0;
////    boolean current = conditions.get(index);
//    for (int f = 0; f < grid.rows(); ++f) {
//      int bin = 0;
//      excluded = 0;
//      count = base[index];
//      ++index;
//      ++bin;
//      while (bin <= grid.row(f).size()) {
//        total += base[index];
//        if (conditions.get(index) == conditions.get(index - 1)) {
//          count += base[index];
//          ++index;
//          ++bin;
//        } else {
//          if (conditions.get(index - 1)) {
//            result -= count * Math.log(count + 1);
//          } else {
//            excluded += count;
//          }
//          realCardinality++;
//          count = base[index];
//          ++index;
//          ++bin;
//        }
//      }
//      if (conditions.get(index - 1)) {
//        result -= count * Math.log(count + 1);
//      } else {
//        excluded += count;
//      }
//      realCardinality++;
//    }

    double weight = 0;
    int index = 0;
    int count = 0;
//    boolean current = conditions.get(index);
    for (int f = 0; f < grid.rows(); ++f) {
      for (int bin = 0; bin < grid.row(f).size(); ++bin, ++index) {
        if (conditions.get(index)) {
          weight += base[index];
        }
      }
    }
    return Math.log(weight + 1);
  }


  private double regularization(BitSet conditions, int[][] base) {
    double result = 0;
    int index = 0;
    int count = 0;
    int realCardinality = 0;
    for (int i = 0; i < base.length; i++) {
      final int[] row = base[i];
      boolean current = conditions.get(index);
      for (int j = 0; j < row.length; j++, index++) {
        realCardinality += (current != conditions.get(index) ? 1 : 0);
        current = conditions.get(index);
        if (!conditions.get(index))
          continue;
        count += row[j];
        if (j + 1 == row.length || !conditions.get(index + 1)) {
          result += Math.log(count + 1);
          count = 0;
        }
      }
    }
    return result + realCardinality; // information in split + AIC
  }
}
