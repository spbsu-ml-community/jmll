package com.expleague.ml.cli.builders.data.impl;

import com.expleague.ml.cli.builders.data.PoolReader;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import java.io.IOException;

public class PoolReaderLetor implements PoolReader {

  @Override
  public Pool read(String path) {
    try {
      return DataTools.loadLetor(path);
    }
    catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
