package com.spbsu.ml.cli.builders.data.impl;

import com.spbsu.ml.cli.builders.data.PoolReader;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;

import java.io.IOException;

/**
 * Created by noxoomo on 15/10/2017.
 */
public class PoolReaderFeatureTxt implements PoolReader {
  @Override
  public Pool read(String path) {
    try {
      return DataTools.loadFromFeaturesTxt(path);
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
