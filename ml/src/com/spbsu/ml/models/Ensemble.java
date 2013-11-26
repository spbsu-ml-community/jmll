package com.spbsu.ml.models;

import com.spbsu.ml.CompositeFunc;
import com.spbsu.ml.Func;
import com.spbsu.ml.VecFunc;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.func.VecFuncJoin;

import java.util.List;

/**
 * User: solar
 * Date: 26.11.13
 * Time: 9:41
 */
public class Ensemble extends CompositeFunc<Linear, VecFunc> {
  public Ensemble(Linear linear, VecFunc vecFunc) {
    super(linear, vecFunc);
  }

  public Ensemble(List<Func> weakModels, double step) {
    this(new Linear(weakModels.size(), step), new VecFuncJoin(weakModels));
  }

  public Ensemble(Func[] models, double[] weights) {
    this(new Linear(weights), new VecFuncJoin(models));
  }

  public int size() {
    return g.ydim();
  }

  public Func[] models() {
    return g.directions();
  }

  public double weight(int i) {
    return f.weights.get(i);
  }

  public Func last() {
    return g.directions()[size() - 1];
  }

  public double wlast() {
    return weight(size() - 1);
  }
}
