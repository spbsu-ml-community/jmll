package com.expleague.ml.data.set;

import com.expleague.commons.math.vectors.Vec;

public interface TimeAwareTargetHolder<Ds extends DataSet> {
  Ds owner();

  Vec target();

  int[] ticks();
}
