package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.BinarizedDataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public abstract class BinOptimizedModel extends Model {
  protected final BFGrid grid;

  protected BinOptimizedModel(BFGrid grid) {
    this.grid = grid;
  }

  public Vec value(DataSet ds) {
    final BinarizedDataSet bds = ds.cache(Binarize.class).binarize(grid);
    Vec result = new ArrayVec(ds.power());
    for (int i = 0; i < ds.power(); i++) {
      result.set(i, value(bds, i));
    }
    return result;
  }

  protected abstract double value(BinarizedDataSet bds, int index);
}
