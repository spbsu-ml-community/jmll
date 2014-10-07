package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;

/**
 * Created by irlab on 07.10.2014.
 */
public interface AttractivenessModel<Blk extends Session.Block> {
  double eval_f(Session<Blk> ses, int s, int e, int click_s);

  SparseVec feats(Session<Blk> ses, int s, int e, int click_s);

  int getEdgeFeatCount();
}
