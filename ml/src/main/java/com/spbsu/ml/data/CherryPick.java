package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;

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
  private final int[] points;
  private final BFGrid grid;
  private final Factory<AdditiveStatistics> factory;

  public CherryPick(BinarizedDataSet bds, Factory<AdditiveStatistics> factory, int[] points) {
    this.bds = bds;
    this.points = points;
    this.grid = bds.grid();
    this.factory = factory;
  }

  private static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("CherryPick thread", -1);

  @SuppressWarnings("unchecked")
  public <T extends AdditiveStatistics> BitSet build(final Evaluator<T> eval, double lambda) {
    int[][] base = new int[grid.rows()][];
    {
      for(int i = 0; i < grid.rows(); i++) {
        final byte[] bin = bds.bins(i);
        base[i] = new int[grid.row(i).size()];
        for(int j = 0; j < points.length; j++) {
          base[i][bin[points[j]]]++;
        }
      }
    }

    final BitSet conditions = new BitSet();
    final T resultStat = (T)factory.create();
    int[] indices2include = points;
    double resultScore = eval.value(resultStat);
    double oldScore;
    do {
      oldScore = resultScore;
      final CountDownLatch latch = new CountDownLatch(grid.size() - conditions.cardinality());
      final double[] scores = new double[grid.size()];
      final int[] candidates = indices2include;
      for (int bfindex = 0; bfindex < grid.rows(); bfindex++) {
        final BFGrid.BinaryFeature bf = grid.bf(bfindex);
        conditions.set(bf.bfIndex);
        scores[bfindex] = lambda * regularization(conditions, base);
        conditions.clear(bf.bfIndex);
        final int finalBfindex = bfindex;
        exec.execute(new Runnable() {
          @Override
          public void run() {
            final byte[] bin = bds.bins(bf.findex);
            final T stat = (T)factory.create();
            for(int i = 0; i < candidates.length; i++) {
              final int index = candidates[i];
              if (bf.binNo == bin[index])
                stat.append(index, 1);
            }
            scores[finalBfindex] += eval.value(stat);
            latch.countDown();
          }
        });
      }
      try {
        latch.await();
      } catch (InterruptedException e) {
        // skip
      }
      final int min = ArrayTools.min(scores);
      if (scores[min] > resultScore)
        break;
      final BFGrid.BinaryFeature bf = grid.bf(min);
      resultScore = scores[min];
      conditions.set(min);
      indices2include = new int[indices2include.length - base[bf.findex][bf.binNo]];
      int index = 0;
      final byte[] bin = bds.bins(bf.findex);
      for(int i = 0; i < candidates.length; i++) {
        if (bin[candidates[i]] == bf.binNo)
          indices2include[index++] = candidates[i];
      }
    }
    while (resultScore > oldScore);
    return conditions;
  }

  private double regularization(BitSet conditions, int[][] base) {
    double result = 0;
    int index = 0;
    int count = 0;
    int realCardinality = 0;
    for(int i = 0; i < base.length; i++) {
      final int[] row = base[i];
      boolean current = conditions.get(index);
      for(int j = 0; j < row.length; j++, index++) {
        realCardinality += (current != conditions.get(index) ? 1 : 0);
        current = conditions.get(index);
        if (!conditions.get(index))
          continue;
        count += row[j];
        if (j+1 == row.length || !conditions.get(index+1)) {
          result += Math.log(count + 1);
          count = 0;
        }
      }
    }
    return result + realCardinality; // information in split + AIC
  }
}
