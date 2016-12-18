package com.spbsu.ml.data.softBorders.dataSet;

import com.spbsu.commons.util.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by noxoomo on 28/11/2016.
 */

public class GreedyBordersSearcher {
  static class Split {
    private final int left;
    private final int right;
    private final int splitIndex;
    private final double score;
    private final double totalWeight;
    private final WeightedFeature feature;

    Split(final int left,
          final int right,
          final WeightedFeature feature) {
      this.left = left;
      this.right = right;
      this.feature = feature;

      double leftWeight = feature.weight(left);
      {
        double tmp = 0;
        for (int j = left; j < right; ++j) {
          tmp += feature.weight(j);
        }
        totalWeight = tmp;
      }

      double bestScore = calcScore(leftWeight, totalWeight - leftWeight);
      int bestSplit = left + 1;
      for (int j = left + 1; j < right; ++j) {
        final double w = feature.weight(j);
        leftWeight += w;
        double sc = calcScore(leftWeight, totalWeight - leftWeight);
        if (sc > bestScore) {
          bestScore = sc;
          bestSplit = j + 1;
        }
      }
      score = bestScore;
      splitIndex = bestSplit;
    }

    boolean canSplit() {
      return (left + 1) < right;
    }

    int lastBinIdx() {
      return right - 1;
    }

    double priority() {
      if (canSplit()) {
        return -score;
      }
      return 0;
    }

    double calcScore(final double leftWeight,
                     final double rightWeight) {
      return Math.log(leftWeight) +  Math.log(rightWeight);
    }

    Pair<Split, Split> split() {
      final Split leftSplit = new Split(left, splitIndex, feature);
      final Split rightSplit = new Split(splitIndex, right, feature);
      return new Pair<>(leftSplit, rightSplit);
    }
  }

  public static int[] borders(final WeightedFeature feature,
                              final int binFactor) {
    PriorityQueue<Split> splits = new PriorityQueue<>((left, right) -> Double.compare(left.priority(), right.priority()));
    splits.add(new Split(0, feature.size(), feature));

    while ((splits.size()) <= binFactor && splits.peek().canSplit()) {
      final Split best = splits.poll();
      if (best.canSplit()) {
        final Pair<Split, Split> tmp = best.split();
        splits.add(tmp.first);
        splits.add(tmp.second);
      }
    }
    List<Split> results = new ArrayList<>(splits);
    Collections.sort(results, (l, r) -> Integer.compare(l.right, r.right));
    return results.subList(0, results.size() - 1).stream().mapToInt(Split::lastBinIdx).toArray();
  }
}
