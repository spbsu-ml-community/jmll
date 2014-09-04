package com.spbsu.ml.cli.builders.data;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.tools.Pool;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public interface DataBuilder extends Factory<Pair<? extends Pool, ? extends Pool>> {
  void setLearnPath(String learnPath);

  void setJsonFormat(boolean isJsonFormat);

  @Override
  Pair<? extends Pool, ? extends Pool> create();
}
