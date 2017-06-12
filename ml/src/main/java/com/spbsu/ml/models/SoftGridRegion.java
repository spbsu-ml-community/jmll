package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.ml.data.impl.BinarizedDataSet;

import java.util.List;

/**
 * User: noxoomo
 */


public class SoftGridRegion extends  BinOptimizedModel.Stub  {
  private final BFGrid.BFRow[] features;
  private final Vec[] cumSumDistributions;
  private final boolean[] mask;
  private final double bias;
  private final double[] values;
  private final BFGrid grid;
  private final FastRandom random = new FastRandom();

  public BFGrid.BFRow[] features() {
    return features.clone();
  }

  public boolean[] masks() {
    return mask.clone();
  }

  public SoftGridRegion(final List<BFGrid.BFRow> features,
                        final List<Vec> distributions,
                        final boolean[] mask,
                        final double bias,
                        final double[] values) {
    this.grid = features.size() > 0 ? features.get(0).grid() : null;
    this.features = features.toArray(new BFGrid.BFRow[features.size()]);
    this.cumSumDistributions = distributions.toArray(new Vec[features.size()]);
    this.mask = mask;
    this.bias = bias;
    this.values = values;
  }

  @Override
  public double value(final BinarizedDataSet bds,
                      final int pindex) {
    double result = bias;

    double p = 1.0;

    for (int i = 0; i < features.length; i++) {
      final int bin = bds.bins(features[i].origFIndex)[pindex];
      final double cumProb = cumSumDistributions[i].get(bin);
      final double currentProb = mask[i] ? cumProb : 1.0 - cumProb;
      result += p * currentProb * values[i];
      p *= currentProb;
    }
    return result;
  }

  public double valueRealization(final BinarizedDataSet bds,
                                 final int pindex) {
    double result = bias;

    for (int i = 0; i < features.length; i++) {
      final byte bin = bds.bins(features[i].origFIndex)[pindex];
      final int binNo = random.nextDiscrete(cumSumDistributions[i]);
      if (bin > binNo != mask[i]) {
        break;
      } else {
        result += values[i];
      }
    }

    return result;
  }
//
  @Override
  public  double value(final Vec x) {
    double result = bias;
    double p = 1.0;

    for (int i = 0; i < features.length; i++) {
      final int bin = features[i].bin(x.get(features[i].origFIndex));
      final double cumProb = cumSumDistributions[i].get(bin);
      final double currentProb = mask[i] ? cumProb : 1.0 - cumProb;
      result += p * currentProb * values[i];
      p *= currentProb;
    }
    return result;
  }



  @Override
  public int dim() {
    return grid.rows();
  }
}
