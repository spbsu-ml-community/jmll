package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.Arrays;

import static java.lang.Math.max;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 12:23
 */
public class MultiLLClassificationLeaf implements BFLeaf {
  private final BFGrid.BFRow[] rows;
  private final LLClassificationLeaf[] folds;
  private final int size;
  private final int classesCount;

  public MultiLLClassificationLeaf(BinarizedDataSet ds, Vec[] point, Vec target, Vec weight) {
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

  private MultiLLClassificationLeaf(BFGrid.BFRow[] rows, LLClassificationLeaf[] folds) {
    this.rows = rows;
    this.folds = folds;
    classesCount = (int)Math.sqrt(folds.length);
    int size = 0;
    for (int i = 0; i < classesCount; i++) {
      size += folds[i * classesCount + i].size();
    }
    this.size = size;
  }

  @Override
  public void append(int feature, byte bin, double target, double current, double weight) {
    for (int c = 0; c < classesCount; c++) { // iterate over instances of this point in LL for all classes
      folds[(int)target * classesCount + c].append(feature, bin, target == c ? 1 : -1, current, weight);
    }
  }

  @Override
  public int score(double[] likelihoods) {
    Vec scores = VecTools.fill(new ArrayVec(likelihoods.length), Double.NEGATIVE_INFINITY);
    for (int f = 0; f < rows.length; f++) {
      LLCounter[] left = new LLCounter[folds.length];
      LLCounter[] right = new LLCounter[folds.length];

      final BFGrid.BFRow row = rows[f];
      {
        for (int i = 0; i < folds.length; i++) {
          left[i] = new LLCounter();
          right[i] = new LLCounter();
          right[i].add(folds[i].total);
        }
      }

      for (int b = 0; b < row.size(); b++) {
        boolean[] mask = new boolean[classesCount];
        for (int i = 0; i < folds.length; i++) {
          left[i].add(folds[i].counter(f, (byte) b));
          right[i].sub(folds[i].counter(f, (byte) b));
        }
        scores.set(row.bfStart + b, max(scores.get(row.bfStart + b),
                optimizeMask(left, mask).score() + optimizeMask(right, mask).score()));
      }
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
    return new MultiLLClassificationLeaf(rows, broLeaves);
  }

  private LLCounter optimizeMask(LLCounter[] folds, boolean[] bestMask) {
    double bestScore = 0;
    Arrays.fill(bestMask, false);
    boolean[] mask = new boolean[bestMask.length];
    final LLCounter combined = new LLCounter();
    for (int i = 0; i < folds.length; i++) {
      add(combined, folds[i], -1.);
    }

    int toAdd;
    while(true) {
      toAdd = -1;
      for (int c = 0; c < mask.length; c++) {
        if (mask[c]) // already in positive examples
          continue;
        for (int t = 0; t < classesCount; t++) {
          final int i = t * classesCount + c;
          sub(combined, folds[i], -1.);
          add(combined, folds[i], 1);
        }
        final double score = combined.score();
        if (bestScore < score) {
          bestScore = score;
          toAdd = c;
        }
        for (int t = 0; t < classesCount; t++) {
          final int i = t * classesCount + c;
          sub(combined, folds[i], 1.);
          add(combined, folds[i], -1);
        }
      }
      if (toAdd >= 0) {
        mask[toAdd] = bestMask[toAdd] = true;
        for (int t = 0; t < mask.length; t++) {
          final int i = t * classesCount + toAdd;
          sub(combined, folds[i], -1);
          add(combined, folds[i], 1);
        }
      }
      else break;
    }
    return combined;
  }

  @Override
  public int size() {
    return size;
  }

  private boolean[] optimalMask;
  @Override
  public double alpha() {
    optimalMask = new boolean[classesCount];
    LLCounter[] classTotals = new LLCounter[folds.length];
    for (int i = 0; i < classTotals.length; i++) {
      classTotals[i] = folds[i].total;
    }
    return optimizeMask(classTotals, optimalMask).alpha();
  }

  public boolean[] mask() {
    if (optimalMask == null)
      alpha();
    return optimalMask;
  }

  /**
   * Combine two parts of $LL=-\sum_{\{a_c\}, b} \sum_c log(1+e^{-1^{I(b=c)}(-a-m(c)x)})$ depending on $m: C \to \{-1,1\}$
   * actually they differs in sign of odd derivations so we need only to properly sum them :)
   */
  private void add(LLCounter combined, LLCounter increment, double sign) {
    combined.maclaurinLL0 += increment.maclaurinLL0;
    combined.maclaurinLL1 += sign * increment.maclaurinLL1;
    combined.maclaurinLL2 += increment.maclaurinLL2;
    combined.maclaurinLL3 += sign * increment.maclaurinLL3;
    combined.maclaurinLL4 += increment.maclaurinLL4;
    combined.bad += increment.bad;
    combined.good += increment.good;
  }

  private void sub(LLCounter combined, LLCounter increment, double sign) {
    combined.maclaurinLL0 -= increment.maclaurinLL0;
    combined.maclaurinLL1 -= sign * increment.maclaurinLL1;
    combined.maclaurinLL2 -= increment.maclaurinLL2;
    combined.maclaurinLL3 -= sign * increment.maclaurinLL3;
    combined.maclaurinLL4 -= increment.maclaurinLL4;
    combined.bad -= increment.bad;
    combined.good -= increment.good;
  }

}
