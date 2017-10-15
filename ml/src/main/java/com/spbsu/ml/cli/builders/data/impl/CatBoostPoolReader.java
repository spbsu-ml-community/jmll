package com.spbsu.ml.cli.builders.data.impl;

import com.spbsu.ml.cli.builders.data.PoolReader;
import com.spbsu.ml.data.tools.CatBoostPoolDescription;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;

import java.io.File;
import java.io.IOException;

/**
 * Created by noxoomo on 15/10/2017.
 */
public class CatBoostPoolReader  implements PoolReader {
  private CatBoostPoolDescription poolDescription;

  public CatBoostPoolReader(final CatBoostPoolDescription poolDescription) {
    this.poolDescription = poolDescription;
  }

  @Override
  public Pool read(String path) {
    try {
      return DataTools.loadFromCatBoostPool(poolDescription, DataTools.gzipOrFileReader(new File(path)));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
