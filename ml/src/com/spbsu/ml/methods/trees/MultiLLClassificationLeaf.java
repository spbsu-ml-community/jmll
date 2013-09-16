package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.MathTools;
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
      LLCounter[] left = new LLCounter[classesCount];
      LLCounter[] right = new LLCounter[classesCount];

      final BFGrid.BFRow row = rows[f];
      {
        for (int c = 0; c < classesCount; c++) {
          left[c] = new LLCounter();
          right[c] = new LLCounter();
          for (int t = 0; t < right.length; t++) {
            right[c].add(folds[t * classesCount + c].total);
          }
        }
      }

      for (int b = 0; b < row.size(); b++) {
        boolean[] mask = new boolean[classesCount];
        for (int c = 0; c < classesCount; c++) {
          for (int t = 0; t < right.length; t++) {
            left[c].add(folds[t * classesCount + c].counter(f, (byte) b));
            right[c].sub(folds[t* classesCount + c].counter(f, (byte) b));
          }
        }
        scores.set(row.bfStart + b, max(scores.get(row.bfStart + b),
                optimizeMaskOpt(left, mask).score() + optimizeMaskOpt(right, mask).score()));
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

  private static LLCounter optimizeMask(LLCounter[] folds, boolean[] mask) {
    double bestScore = 0;
    int classesCount = mask.length;
    Arrays.fill(mask, false);
    final LLCounter combined = new LLCounter();
    for (int i = 0; i < folds.length; i++) {
      combined.add(folds[i], -1.);
    }

    int toAdd;
    while(true) {
      toAdd = -1;
      for (int c = 0; c < mask.length; c++) {
        if (mask[c]) // already in positive examples
          continue;
        for (int t = 0; t < classesCount; t++) {
          final int i = t * classesCount + c;
          combined.sub(folds[i], -1.);
          combined.add(folds[i], 1.);
        }
        final double score = combined.score();
        if (bestScore < score) {
          bestScore = score;
          toAdd = c;
        }
        for (int t = 0; t < classesCount; t++) {
          final int i = t * classesCount + c;
          combined.sub(folds[i], 1.);
          combined.add(folds[i], -1.);
        }
      }
      if (toAdd >= 0) {
        mask[toAdd]  = true;
        for (int t = 0; t < mask.length; t++) {
          final int i = t * classesCount + toAdd;
          combined.sub(folds[i], -1.);
          combined.add(folds[i], 1.);
        }
      }
      else break;
    }
    return combined;
  }

  private static LLCounter optimizeMaskOpt(LLCounter[] folds, boolean[] mask) {
    double bestScore = 0;
    Arrays.fill(mask, false);
    final int classesCount = mask.length;
    final LLCounter combined = new LLCounter();
    for (int i = 0; i < classesCount; i++) {
      combined.add(folds[i], -1.);
    }

    int toAdd;
    while(true) {
      toAdd = -1;
      for (int c = 0; c < classesCount; c++) {
        if (mask[c]) // already in positive examples
          continue;
        combined.sub(folds[c], -1.);
        combined.add(folds[c], 1.);
        final double score = combined.score();
        if (bestScore < score) {
          bestScore = score;
          toAdd = c;
        }
        combined.sub(folds[c], 1.);
        combined.add(folds[c], -1.);
      }
      if (toAdd >= 0) {
        mask[toAdd] = mask[toAdd] = true;
        combined.sub(folds[toAdd], -1.);
        combined.add(folds[toAdd], 1.);
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
    boolean[] optimalMask = new boolean[classesCount];
    boolean[] optimalMaskOpt = new boolean[classesCount];
    LLCounter[] classTotals = new LLCounter[folds.length];
    for (int i = 0; i < classTotals.length; i++) {
      classTotals[i] = folds[i].total;
    }
    LLCounter[] classTotalsOpt = new LLCounter[classesCount];
    for (int c = 0; c < classesCount; c++) {
      classTotalsOpt[c] = new LLCounter();
      for (int t = 0; t < classesCount; t++) {
        classTotalsOpt[c].add(folds[t * classesCount + c].total);
      }
    }
    final double alpha1 = optimizeMask(classTotals, optimalMask).alpha();
    final double alpha2 = optimizeMaskOpt(classTotalsOpt, optimalMaskOpt).alpha();
    if (Math.abs(alpha1 - alpha2) > MathTools.EPSILON
            || !Arrays.equals(optimalMask, optimalMaskOpt)) {
      if (Math.abs(alpha1 + alpha2) < MathTools.EPSILON) {
        for (int i = 0; i < optimalMaskOpt.length; i++) {
          if (optimalMaskOpt[i] == optimalMask[i]) {
            System.out.println();
          }
        }
      }
      else System.out.println();
    }
    double alpha = optimizeMask(classTotals, this.optimalMask).alpha();
//    if (Math.abs(optimizeMask(classTotals, optimalMask).alpha() + optimizeMaskOpt(classTotalsOpt, optimalMaskOpt).alpha()) < MathTools.EPSILON) {
//      for (int i = 0; i < this.optimalMask.length; i++) {
//        this.optimalMask[i] = !this.optimalMask[i];
//      }
//      alpha = -alpha;
//    }
    return alpha;
  }

  public boolean[] mask() {
    if (optimalMask == null)
      alpha();
    return optimalMask;
  }

}
