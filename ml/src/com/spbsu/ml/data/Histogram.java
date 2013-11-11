package com.spbsu.ml.data;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;

/**
* User: solar
* Date: 05.12.12
* Time: 21:19
*/
public class Histogram implements Aggregator {
  BFGrid grid;
  private final Vec weight;
  private final Vec target;
  int maxBins = 0;

  double[] sums;
  double[] sums2;
  double[] weights;

  public Histogram(BFGrid grid, Vec weight, Vec target) {
    this.grid = grid;
    this.weight = weight;
    this.target = target;
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
  public void append(int feature, byte bin, int index) {
    int findex = feature * maxBins + bin;
    sums[findex] += target.get(index);
    sums2[findex] += target.get(index) * target.get(index);
    weights[findex] += weight.get(index);
  }
}
