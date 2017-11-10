package com.expleague.ml.data.set.impl;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.Vectorization;
import com.expleague.ml.meta.DSItem;

import java.util.stream.BaseStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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

  @Override
  public Seq<Vec> sub(int start, int end) {
    return new VecDataSetImpl(data.sub(start, 0, end - start, data.columns()), (VecDataSet)parent());
  }

  @Override
  public Stream<Vec> stream() {
    return IntStream.range(0, length()).mapToObj(i -> data().row(i));
  }
}
