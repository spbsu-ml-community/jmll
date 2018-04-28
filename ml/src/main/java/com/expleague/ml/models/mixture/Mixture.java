package com.expleague.ml.models.mixture;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import org.jetbrains.annotations.NotNull;

public interface Mixture extends Trans {
  default double prob(Vec sample) {
    return trans(sample).get(0);
  }
  Vec probAll(Mx samples, Vec to);

  Vec byComponentProb(Vec sample, Vec to);
  Mx byComponentProb(Mx samples, Mx to);

  @Override
  default Vec transTo(Vec x, Vec to) {
    return byComponentProb(x, to);
  }

  @Override
  default Mx transAll(Mx ds) {
    Mx to = new VecBasedMx(ds.rows(), numComponents());
    return byComponentProb(ds, to);
  }

  void setParameters(Vec parameters);

  @NotNull
  Vec getParameters();
  void upgradeParameters(Mx probBySample, Mx samples);

  int numComponents();
  int wdim();
}
