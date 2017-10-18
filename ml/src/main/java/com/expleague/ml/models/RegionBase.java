package com.expleague.ml.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.BinOptimizedModel;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.BFGrid;

/**
 * User: solar
 * Date: 05.12.14
 * Time: 20:15
 */
public abstract class RegionBase extends BinOptimizedModel.Stub {
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
}

