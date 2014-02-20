package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Func;

/**
 * Created by towelenee on 20.02.14.
 */
public class PolynomialExponentRegion extends Func.Stub {
  private final BFGrid.BinaryFeature[] features;
  private final boolean[] mask;
  private final double[] value;
  private final double distCoeffiecent;

  public PolynomialExponentRegion(BFGrid.BinaryFeature[] features, boolean[] mask, double[] value, double distCoeffiecent) {
    this.features = features;
    this.mask = mask;
    this.value = value;
    this.distCoeffiecent = distCoeffiecent;
  }

  double getDistanseFromRegion(Vec x) {
    double distanse = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        distanse += Math.pow(features[i].condition - x.get(features[i].findex), 2);
    }
    return distanse;
  }

  @Override
  public double value(Vec x) {
    double data[] = new double[features.length + 1];
    double ans = 0;
    data[0] = 1;
    for(int i = 0; i < features.length;i++)
      data[i + 1] = x.get(features[i].findex);
    for(int i = 0; i <= features.length; i++)
      for(int j = 0; j <= features.length; j++)
        ans += data[i] * data[j] * value[i + j * (features.length + 1)];
    return ans * Math.exp(- distCoeffiecent * getDistanseFromRegion(x)) ;
  }

  @Override
  public int dim() {
    return features[0].row().grid().rows();
  }
}
