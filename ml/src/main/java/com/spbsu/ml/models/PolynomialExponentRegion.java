package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;
import com.spbsu.commons.math.Func;

/**
 * Created by towelenee on 20.02.14.
 */
public class PolynomialExponentRegion extends Func.Stub {
  private final BFGrid.BinaryFeature[] features;
  private final boolean[] mask;
  private final double[] value;
  private final double distCoeffiecent;

  public PolynomialExponentRegion(final BFGrid.BinaryFeature[] features, final boolean[] mask, final double[] value, final double distCoeffiecent) {
    this.features = features;
    this.mask = mask;
    this.value = value;
    this.distCoeffiecent = distCoeffiecent;
  }

  double getDistanseFromRegion(final Vec x) {
    final double distanse = 0;
    for (int i = 0; i < features.length; i++) {
      if (features[i].value(x) != mask[i])
        return 0;
        //distanse += Math.pow(lines[i].condition - x.at(lines[i].findex), 2);
    }
    //return distanse;
    return 1;
  }

  @Override
  public double value(final Vec x) {
    final double[] data = new double[features.length + 1];
    double ans = 0;
    data[0] = 1;
    for(int i = 0; i < features.length;i++)
      data[i + 1] = x.get(features[i].findex);
    for(int i = 0; i <= features.length; i++)
      for(int j = 0; j <= features.length; j++)
        ans += data[i] * data[j] * value[i + j * (features.length + 1)];
    return ans * getDistanseFromRegion(x);
  }

  @Override
  public int dim() {
    return features[0].row().grid().rows();
  }
}
