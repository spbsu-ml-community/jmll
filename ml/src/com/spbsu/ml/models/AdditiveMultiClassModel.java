package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.MultiClassModel;

import java.util.Iterator;
import java.util.List;

/**
 * User: solar
 * Date: 10.09.13
 * Time: 15:52
 */
public class AdditiveMultiClassModel<T extends MultiClassModel> extends MultiClassModel {
  public final List<T> models;
  public final double step;

  public AdditiveMultiClassModel(List<T> models, double step) {
    this.models = models;
    this.step = step;
  }

  public double value(Vec point) {
    Iterator<T> iter = models.iterator();
    double result = 0;
    while (iter.hasNext()) {
      result += step * iter.next().value(point);
    }
    return result;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof AdditiveModel)) return false;

    AdditiveModel that = (AdditiveModel) o;

    return Double.compare(that.step, step) == 0 && models.equals(that.models);

  }

  @Override
  public int hashCode() {
    int result;
    long temp;
    result = models.hashCode();
    temp = Double.doubleToLongBits(step);
    result = 31 * result + (int) (temp ^ (temp >>> 32));
    return result;
  }

  @Override
  public double value(Vec point, int classNo) {
    double result = 0;
    for (T model : models) {
      result += step * model.value(point, classNo);
    }
    return result;
  }
}
