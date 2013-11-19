package com.spbsu.ml;

import com.spbsu.commons.func.WeakListenerHolder;
import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface CursorOracle<LocalLoss extends Oracle0> extends Oracle0, WeakListenerHolder<CursorOracle.CursorMoved> {
  interface CursorMoved {
    void to(Vec v);
  }
  Vec cursor();
  Vec moveTo(Vec m);

  LocalLoss local();
}
