package com.spbsu.ml.data;

import com.spbsu.commons.func.CacheHolder;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:22
 */
public interface DataSet extends CacheHolder {
  int power();
  int xdim();

  DSIterator iterator();

  Mx data();
  Vec target();
}
