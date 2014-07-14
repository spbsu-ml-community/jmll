package com.spbsu.ml.data.set.impl;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.ml.Vectorization;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.meta.DSItem;

/**
 * User: solar
 * Date: 09.07.14
 * Time: 21:10
 */
public class VecDataSetImpl extends VecDataSet.Stub {
  private final Mx data;

  public VecDataSetImpl(final Mx data, final VecDataSet parent) {
    super(parent);
    this.data = data;
  }

  public <I extends DSItem> VecDataSetImpl(final DataSet<I> parent, final Mx data, final Vectorization<I> vec) {
    super(parent, vec);
    this.data = data;
  }

  @Override
  public Mx data() {
    return data;
  }
}
