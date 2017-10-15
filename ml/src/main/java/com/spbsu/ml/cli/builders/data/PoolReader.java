package com.spbsu.ml.cli.builders.data;

import com.spbsu.ml.data.tools.Pool;

import java.io.IOException;

/**
 * Created by noxoomo on 15/10/2017.
 */
public interface PoolReader {
  Pool read(final String path);
}

