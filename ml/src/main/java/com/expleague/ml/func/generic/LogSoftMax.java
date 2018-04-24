package com.expleague.ml.func.generic;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

import static java.lang.Math.exp;

public class LogSoftMax extends FuncC1.Stub {
  private final int nClasses;
  private final int trueClass;

  public LogSoftMax(int nClasses, int trueClass) {
    this.nClasses = nClasses;
    this.trueClass = trueClass;
  }

  public static double sumExp(Vec argument) {
    double sumExp = 0.;
    for (int i = 0; i < argument.dim(); i++) {
      sumExp += exp(argument.get(i));
    }

    return sumExp;
  }

  @Override
  public Vec gradientTo(Vec x, Vec to) {
    final double sumExp = sumExp(x);
    VecTools.assign(to, x);
    for (int i = 0; i < to.dim(); i++) {
      to.set(i, Math.exp(to.get(i)) / sumExp);
    }
    to.adjust(trueClass, -1.);
    return to;
  }

  public static double staticValue(Vec x, int trueClass) {
    return - x.get(trueClass) + Math.log(sumExp(x));
  }

  @Override
  public double value(Vec x) {
    return staticValue(x, trueClass);
  }

  @Override
  public int dim() {
    return nClasses;
  }
}
