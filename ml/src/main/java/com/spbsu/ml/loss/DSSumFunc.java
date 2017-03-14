package com.spbsu.ml.loss;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 12:52
 */
public abstract class DSSumFunc<Item> extends Func.Stub implements TargetFunc {
  protected final DataSet<Item> ds;

  public DSSumFunc(DataSet<Item> ds) {
    this.ds = ds;
  }

  public abstract Func component(int index);

  public Item item(int index) {
    return ds.at(index);
  }

  public int length() {
    return ds.length();
  }

  @Override
  public double value(Vec x) {
    final int length = length();
    double result = 0;
    for (int i = 0; i < length; i++) {
      result += component(i).value(x);
    }
    return result;
  }

  @Override
  public int dim() {
    return component(0).dim();
  }


  @Override
  public DataSet<?> owner() {
    return ds;
  }
}
