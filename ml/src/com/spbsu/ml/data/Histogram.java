package com.spbsu.ml.data;

import com.spbsu.ml.BFGrid;

/**
* User: solar
* Date: 05.12.12
* Time: 21:19
*/
public class Histogram implements Aggregator {
  BFGrid grid;
  int maxBins = 0;

  double[] sums;
  double[] sums2;
  double[] weights;

  public Histogram(BFGrid grid) {
    this.grid = grid;
    for (int f = 0; f < grid.rows(); f++) {
      maxBins = Math.max(maxBins, grid.row(f).size() + 1);
    }
    sums = new double[maxBins * grid.rows()];
    sums2 = new double[maxBins * grid.rows()];
    weights = new double[maxBins * grid.rows()];
  }

  public interface Judge {
    double score(double sum, double sum2, double weight, int bf);
  }

  public void score(double[] scores, Judge judge) {
    for (int findex = 0; findex < grid.rows(); findex++) {
      final BFGrid.BFRow row = grid.row(findex);
      final int bfStart = findex * maxBins;
      double sum = 0;
      double sum2 = 0;
      double weight = 0;
      for (int bin = 0; bin < row.size(); bin++) {
        sum += sums[bfStart + bin];
        sum2 += sums2[bfStart + bin];
        weight += weights[bfStart + bin];
        scores[row.bfStart + bin] += judge.score(sum, sum2, weight, row.bfStart + bin);
      }
    }
  }

  @Override
  public void append(int feature, byte bin, double target, double current, double weight) {
    int index = feature * maxBins + bin;
    sums[index] += target;
    sums2[index] += target * target;
    weights[index] += weight;
  }
}
