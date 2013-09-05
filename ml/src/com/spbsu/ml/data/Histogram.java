package com.spbsu.ml.data;

import com.spbsu.ml.BFGrid;
import gnu.trove.TDoubleDoubleProcedure;

/**
* User: solar
* Date: 05.12.12
* Time: 21:19
*/
public class Histogram implements Aggregator {
  BFGrid grid;
  double[] sums;
  double[] weights;

  public Histogram(BFGrid grid) {
    this.grid = grid;
    sums = new double[grid.size() + grid.rows()];
    weights = new double[grid.size() + grid.rows()];
  }

  public void set(int feature, int bin, double sum, double weight) {
    if (bin > 0) {
      final BFGrid.BinaryFeature bf = grid.row(feature).bf(bin - 1);
      sums[bf.bfIndex] = sum;
      weights[bf.bfIndex] = weight;
    }
  }

  public void process(int bfeature, TDoubleDoubleProcedure procedure) {
    final BFGrid.BinaryFeature bf = grid.bf(bfeature);
    final BFGrid.BFRow row = bf.row();
    double sum = 0;
    double weight = 0;
    for (int bfindex = bfeature; bfindex < row.bfEnd; bfindex++) {
      sum += sums[bfindex];
      weight += weights[bfindex];
    }
    procedure.execute(weight, sum);
  }

  @Override
  public void append(int feature, byte bin, double target, double current, double weight) {
    if (bin > 0) {
      final BFGrid.BinaryFeature bf = grid.row(feature).bf(bin - 1);
      sums[bf.bfIndex] += target;
      weights[bf.bfIndex] += weight;
    }
  }
}
