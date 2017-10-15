package com.spbsu.ml.cli.builders.data;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.tools.Pool;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public interface DataBuilder extends Factory<Pair<? extends Pool, ? extends Pool>> {

  void setLearnPath(final String learnPath);
  void setReader(final PoolReader reader);

  @Override
  Pair<? extends Pool, ? extends Pool> create();
}
