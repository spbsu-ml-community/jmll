package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static java.lang.Math.max;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:23
 */
public class MultiLLClassificationLeaf implements BFLeaf {
  public static final double THRESHOLD = 3.;
  private final BFGrid.BFRow[] rows;
  private final LLClassificationLeaf[] folds;
  private final int size;
  private final int classesCount;
  private final Vec[] point;
  private final Vec target;
  private final Vec weight;

  public MultiLLClassificationLeaf(BinarizedDataSet ds, Vec[] point, Vec target, Vec weight) {
    this.point = point;
    this.target = target;
    this.weight = weight;
    classesCount = point.length;

    folds = new LLClassificationLeaf[classesCount * classesCount];
    Vec[] classTargets = new Vec[classesCount];
    int[][] classIndices = new int[classesCount][];
    for (int c = 0; c < classesCount; c++) {
      classIndices[c] = DataTools.extractClass(target, c);
      classTargets[c] = VecTools.fillIndices(VecTools.fill(new ArrayVec(target.dim()), -1.), classIndices[c], 1);
    }

    for (int c = 0; c < classesCount; c++) {
      for (int t = 0; t < classesCount; t++) {
        folds[c * classesCount + t] = new LLClassificationLeaf(ds, point[t], classTargets[t], weight, classIndices[c]);
      }
    }
    rows = ds.grid().allRows();
    size = ds.original().power();
  }

  private MultiLLClassificationLeaf(BFGrid.BFRow[] rows, LLClassificationLeaf[] folds, Vec[] point, Vec target, Vec weight) {
    this.rows = rows;
    this.folds = folds;
    this.point = point;
    this.target = target;
    this.weight = weight;
    classesCount = (int)Math.sqrt(folds.length);
    int size = 0;
    for (int i = 0; i < classesCount; i++) {
      size += folds[i * classesCount + i].size();
    }
    this.size = size;
  }

  @Override
  public void append(int feature, byte bin, int index) {
    final int classId = (int) target.get(index);
    for (int c = 0; c < classesCount; c++) { // iterate over instances of this point in LL for all classes
      folds[classId * classesCount + c].append(feature, bin, index);
    }
  }

  private static ThreadPoolExecutor executor;
  private static synchronized ThreadPoolExecutor executor(int queueSize) {
    if (executor == null || executor.getQueue().remainingCapacity() < queueSize) {
      executor = ThreadTools.createExecutor("MultiLLClassificationLeaf optimizer", queueSize);
    }
    return executor;
  }

  @Override
  public int score(double[] likelihoods) {
    final Vec scores = VecTools.fill(new ArrayVec(likelihoods.length), Double.NEGATIVE_INFINITY);
    final ThreadPoolExecutor exe = executor(rows.length);
    final CountDownLatch latch = new CountDownLatch(rows.length);
    for (int f = 0; f < rows.length; f++) {
      final int finalF = f;
      exe.execute(new Runnable() {
        @Override
        public void run() {
          final BFGrid.BFRow row = rows[finalF];

          LLCounter[] left = new LLCounter[classesCount];
          LLCounter[] right = new LLCounter[classesCount];
          LLCounter[] agg = new LLCounter[classesCount * row.size()];

          {
            for (int c = 0; c < classesCount; c++) {
              left[c] = new LLCounter();
              right[c] = new LLCounter();
              for (int t = 0; t < right.length; t++) {
                right[c].add(folds[t * classesCount + c].total);
              }
              for (int b = 0; b < row.size(); b++) {
                agg[c * row.size() + b] = new LLCounter();
                for (int t = 0; t < right.length; t++) {
                  agg[c * row.size() + b].add(folds[t * classesCount + c].counter(finalF, (byte)b));
                }
              }
            }
          }

          for (int b = 0; b < row.size(); b++) {
            boolean[] mask = new boolean[classesCount];
            for (int c = 0; c < classesCount; c++) {
              left[c].add(agg[c * row.size() + b]);
              right[c].sub(agg[c * row.size() + b]);
            }
            scores.set(row.bfStart + b, max(scores.get(row.bfStart + b),
                    optimizeMask(left, mask).score() + optimizeMask(right, mask).score()));
          }
          latch.countDown();
        }
      });
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      //
    }
    for (int i = 0; i < likelihoods.length; i++) {
      likelihoods[i] += scores.get(i);
    }
    return ArrayTools.max(scores.toArray());
  }

  @Override
  public MultiLLClassificationLeaf split(BFGrid.BinaryFeature feature) {
    LLClassificationLeaf[] broLeaves = new LLClassificationLeaf[folds.length];
    for (int i = 0; i < broLeaves.length; i++) {
      broLeaves[i] = folds[i].split(feature);
    }
    return new MultiLLClassificationLeaf(rows, broLeaves, point, target, weight);
  }

  private static LLCounter optimizeMask(LLCounter[] folds, boolean[] mask) {
    final int classesCount = mask.length;
    final boolean[] confidence = new boolean[classesCount];
    double denom = 0;
    for (int c = 0; c < classesCount; c++) {
      denom += folds[c].d2;
    }
    final LLCounter combined = new LLCounter();
    double maxLL = 0;
    for (int c = 0; c < classesCount; c++) {
      combined.add(folds[c]);
      maxLL = max(folds[c].ll, maxLL);
    }
    int bad = 0;
    for (int c = 0; c < classesCount; c++) {
//      if (abs(folds[c].d1 * folds[c].d1/denom) > 10) {
//        confidence[c] = true;
//      }
      mask[c] = folds[c].d1 > 0;
      if (!mask[c]) {
        bad += folds[c].good;
        combined.invert(folds[c], -1);
      }
    }
    if (combined.good < 2) {
//      System.out.println();
      return combined;
    }
    combined.bad = bad;
    combined.good -= combined.bad;
    double bestScore = combined.score();
    int best;
    while(true) {
      best = -1;
      for (int c = 0; c < classesCount; c++) {
        if (confidence[c]) // already in positive examples or definite negative
          continue;
        combined.invert(folds[c], mask[c] ? -1. : 1.);
        final double score = combined.score();//maxLL);
        if (bestScore < score) {
          bestScore = score;
          best = c;
        }
        combined.invert(folds[c], mask[c] ? 1. : -1.);
      }
      if (best >= 0) {
        confidence[best] = true;
        combined.invert(folds[best], mask[best] ? -1. : 1.);
      }
      else break;
    }
//    boolean[] fullMask = new boolean[classesCount];
//    final double fullScore = optimizeMaskFull(folds, fullMask).score();
//    if (fullScore - combined.score() > 0.1) {
//      synchronized (System.class) {
//        for (int c = 0; c < classesCount; c++) {
//          System.out.println(abs(folds[c].d1 * folds[c].d1/denom) + " " + folds[c].d1 + " " + fullMask[c] + " " + mask[c] + " " + confidence[c]);
//        }
//        System.out.println();
//      }
//    }
    return combined;
  }

  @Override
  public int size() {
    return size;
  }

  private boolean[] optimalMask;
  @Override
  public double alpha() {
    this.optimalMask = new boolean[classesCount];
    return optimize(this.optimalMask).alpha();
  }

  private LLCounter optimize(boolean[] mask) {
    LLCounter[] classTotalsOpt = new LLCounter[classesCount];
    for (int c = 0; c < classesCount; c++) {
      classTotalsOpt[c] = new LLCounter();
      for (int t = 0; t < classesCount; t++) {
        classTotalsOpt[c].add(folds[t * classesCount + c].total);
      }
    }
    return optimizeMask(classTotalsOpt, mask);
  }

  public boolean[] mask() {
    if (optimalMask == null)
      alpha();
    return optimalMask;
  }

  public double score() {
    return optimize(new boolean[classesCount]).score();
  }
}
