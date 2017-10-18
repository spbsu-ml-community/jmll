package com.expleague.ml.cli.builders.data;

import com.expleague.ml.data.tools.Pool;
import com.expleague.commons.func.Factory;
import com.expleague.commons.util.Pair;

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
