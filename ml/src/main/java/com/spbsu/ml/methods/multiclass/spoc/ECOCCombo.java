package com.spbsu.ml.methods.multiclass.spoc;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.IndexTransVec;
import com.spbsu.commons.util.Combinatorics;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.MulticlassCodingMatrixModel;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntObjectMap;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.*;

/**
 * User: qdeee
 * Date: 14.11.14
 */
public class ECOCCombo extends WeakListenerHolderImpl<MulticlassCodingMatrixModel> implements VecOptimization<BlockwiseMLLLogit> {
  private static final int UNITS = Runtime.getRuntime().availableProcessors();
  private static final double MX_IGNORE_THRESHOLD = 0.1;
  private final ExecutorService executor;

  protected final int k;
  protected final int l;
  protected final double lambdaC;
  protected final double lambdaR;
  protected final double lambda1;
  protected final Mx S;

  protected final VecOptimization<LLLogit> weak;

  public ECOCCombo(
      final int k,
      final int l,
      final double lambdaC,
      final double lambdaR,
      final double lambda1,
      final @NotNull Mx s,
      final @NotNull VecOptimization<LLLogit> weak
  ) {
    this.executor = Executors.newFixedThreadPool(UNITS);
    this.k = k;
    this.l = l;
    this.lambdaC = lambdaC;
    this.lambdaR = lambdaR;
    this.lambda1 = lambda1;
    this.S = s;
    this.weak = weak;
  }

  protected double calcLoss(final Mx B, final Mx S) {
    double result = 0;
    final Mx mult = MxTools.multiply(B, MxTools.transpose(B));
    result -= MxTools.trace(MxTools.multiply(mult, S));
    result += lambdaR * VecTools.sum(mult);
    result += lambdaC * VecTools.sum2(B);
    result += lambda1 * VecTools.l1(B);
    return result;
  }

  @Override
  public Trans fit(final VecDataSet learn, final BlockwiseMLLLogit mllLogit) {
    final long permutationsForOneProcessor = (long)(Math.pow(2, k) + UNITS - 1) / UNITS;
    final Mx mxB = new VecBasedMx(k, l);
    final List<Func> classifiers = new ArrayList<>(l);
    final TIntObjectMap<TIntList> indexes = MCTools.splitClassesIdxs(mllLogit.labels());

    for (int j = 0; j < l; j++) {
      //find column
      final List<Callable<Pair<Double, int[]>>> tasks = new LinkedList<Callable<Pair<Double, int[]>>>();
      for (int u = 0; u < UNITS; u++) {
        final Mx mxBCopy = VecTools.copy(mxB.sub(0, 0, k, j + 1));
        final long start = u * permutationsForOneProcessor;
        tasks.add(new ColumnSearch(S, mxBCopy, start, permutationsForOneProcessor));
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
      } catch (InterruptedException | ExecutionException e) {
        e.printStackTrace(); //who cares?
      }

      //fit column classifier
      final TIntList learnIdxs = new TIntLinkedList();
      final TDoubleList target = new TDoubleLinkedList();
      for (int i = 0; i < k; i++) {
        final double code = mxB.get(i, j);
        if (Math.abs(code) > MX_IGNORE_THRESHOLD) {
          final TIntList classIdxs = indexes.get(i);
          target.fill(target.size(), target.size() + classIdxs.size(), Math.signum(code));
          learnIdxs.addAll(classIdxs);
        }
      }

      final VecDataSet dataSet = new VecDataSetImpl(
          new VecBasedMx(
              learn.xdim(),
              new IndexTransVec(
                  learn.data(),
                  new RowsPermutation(learnIdxs.toArray(), learn.xdim())
              )
          ), learn
      );
      final LLLogit loss = new LLLogit(new ArrayVec(target.toArray()), learn);
      classifiers.add((Func) weak.fit(dataSet, loss));

      invoke(
          new MulticlassCodingMatrixModel(
            mxB.sub(0, 0, k, j + 1),
            classifiers.toArray(new Func[j + 1]),
            MX_IGNORE_THRESHOLD)
      );
    }
    executor.shutdown();

    return new MulticlassCodingMatrixModel(mxB, classifiers.toArray(new Func[classifiers.size()]), MX_IGNORE_THRESHOLD);
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
}
