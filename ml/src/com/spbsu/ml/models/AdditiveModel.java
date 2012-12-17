package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Model;

import java.util.Iterator;
import java.util.List;

/**
* User: solar
* Date: 26.11.12
* Time: 15:56
*/
public class AdditiveModel implements Model {
  public final List<Model> models;
  public final double step;

  public AdditiveModel(List<Model> models, double step) {
    this.models = models;
    this.step = step;
  }

  public double value(Vec point) {
    Iterator<Model> iter = models.iterator();
    double result = 0;
    while (iter.hasNext()) {
      result += step * iter.next().value(point);
    }
    return result;
  }
}
