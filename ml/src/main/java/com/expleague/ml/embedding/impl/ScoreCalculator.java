package com.expleague.ml.embedding.impl;

import java.util.Arrays;

public class ScoreCalculator {
  private final double[] scores;
  private final double[] weights;
  private final long[] counts;

  public ScoreCalculator(int dim) {
    counts = new long[dim];
    scores = new double[dim];
    weights = new double[dim];
  }

  public void adjust(int i, int j, double weight, double value) {
    weights[i] += weight;
    scores[i] += value;
    counts[i] ++;
  }

  public double gloveScore() {
    return Arrays.stream(scores).sum() / Arrays.stream(counts).sum();
  }

  public long count() {
    return Arrays.stream(counts).sum();
  }
}
