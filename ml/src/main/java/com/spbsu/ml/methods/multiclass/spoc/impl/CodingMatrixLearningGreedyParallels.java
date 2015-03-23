package com.spbsu.ml.methods.multiclass.spoc.impl;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.util.Combinatorics;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.methods.multiclass.spoc.CMLHelper;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.*;

/**
 * User: qdeee
 * Date: 05.06.14
 */
public class CodingMatrixLearningGreedyParallels extends CodingMatrixLearningGreedy {
  private final ThreadPoolExecutor executor;
  private final int units;

  public CodingMatrixLearningGreedyParallels(final int k, final int l, final double lambdaC, final double lambdaR, final double lambda1) {
    super(k, l, lambdaC, lambdaR, lambda1);
    units = Runtime.getRuntime().availableProcessors();
    executor = new ThreadPoolExecutor(units, units, 5, TimeUnit.DAYS, new LinkedBlockingDeque<Runnable>());
  }

  private class ColumnSearch implements Callable<Pair<Double, int[]>> {
    final Mx mxB;
    final Mx S;
    final long start;
    final long count;

    private ColumnSearch(final Mx S, final Mx mxB, final long start, final long count) {
      this.mxB = mxB;
      this.start = start;
      this.count = count;
      this.S = S;
    }

    @Override
    public Pair<Double, int[]> call() throws Exception {
      final Combinatorics.Enumeration generator = new Combinatorics.PartialPermutations(2, mxB.rows());
      generator.skipN(start);

      double minLoss = Double.MAX_VALUE;
      int[] bestPerm = null;
      int pos = 0;
      while (pos++ < count && generator.hasNext()) {
        final int[] perm = generator.next();
        for (int i = 0; i < mxB.rows(); i++) {
          mxB.set(i, mxB.columns() - 1, 2 * perm[i] - 1);  //0 -> -1, 1 -> 1
        }
        if (CMLHelper.checkConstraints(mxB) && CMLHelper.checkColumnsIndependence(mxB)) {
          final double loss = calcLoss(mxB, S);
          if (loss < minLoss) {
            minLoss = loss;
            bestPerm = perm;
          }
        }
      }
      if (bestPerm == null) {
        throw new IllegalStateException("Not found appreciate column #" + (mxB.columns() - 1));
      }
      return Pair.create(minLoss, bestPerm);
    }
  }

  @Override
  public Mx findMatrixB(final Mx S) {
    final long partition = (long)(Math.pow(2, k) + units - 1) / units;
    final Mx mxB = new VecBasedMx(k, l);
    for (int j = 0; j < l; j++) {
      final List<Callable<Pair<Double, int[]>>> tasks = new LinkedList<Callable<Pair<Double, int[]>>>();
      for (int u = 0; u < units; u++) {
        final Mx mxBCopy = VecTools.copy(mxB.sub(0, 0, k, j + 1));
        final long start = u * partition;
        tasks.add(new ColumnSearch(S, mxBCopy, start, partition));
      }
      try {
        final List<Future<Pair<Double,int[]>>> futures = executor.invokeAll(tasks);
        double totalMinLoss = Double.MAX_VALUE;
        int[] totalBestPerm = null;
        for (final Future<Pair<Double, int[]>> future : futures) {
          final Pair<Double, int[]> pair = future.get();
          final Double loss = pair.first;
          final int[] perm = pair.second;

          if (loss < totalMinLoss) {
            totalMinLoss = loss;
            totalBestPerm = perm;
          }
        }
        for (int i = 0; i < totalBestPerm.length; i++) {
          mxB.set(i, j, 2 * totalBestPerm[i] - 1);
        }
      } catch (InterruptedException e) {
        e.printStackTrace(); //who cares?
      } catch (ExecutionException e) {

      }
      System.out.println("Column " + j + " is over!");
    }
    return mxB;
  }


}
