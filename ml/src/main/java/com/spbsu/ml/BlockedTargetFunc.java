package com.spbsu.ml;

import com.spbsu.commons.math.Func;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 15:05
 */
public interface BlockedTargetFunc extends TargetFunc {
  Func block(int index);
  int blocksCount();
}
