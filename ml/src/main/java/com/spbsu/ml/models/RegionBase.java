package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.BinOptimizedModel;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.impl.BinarizedDataSet;

/**
 * User: solar
 * Date: 05.12.14
 * Time: 20:15
 */
public abstract class RegionBase extends Func.Stub implements BinOptimizedModel {
  public final double inside;
  public final double outside;
  public final BFGrid grid;

  public RegionBase(final BFGrid grid, final double inside, final double outside) {
    this.grid = grid;
    this.inside = inside;
    this.outside = outside;
  }

  @Override
  public final double value(final BinarizedDataSet bds, final int pindex) {
    return contains(bds, pindex) ? inside : outside;
  }

  @Override
  public final double value(final Vec x) {
    return contains(x)? inside : outside;
  }

  @Override
  public final int dim() {
    return grid.rows();
  }


  public abstract boolean contains(BinarizedDataSet bds, int pindex);

  public abstract boolean contains(Vec x);

  public final Vec transAll(BinarizedDataSet bds) {
    Vec result = new ArrayVec(bds.original().length());
    for (int i=0; i < result.dim();++i)
      result.set(i,value(bds,i));
    return result;
  }


}

