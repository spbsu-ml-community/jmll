package com.expleague.ml.cli.builders.data;


import com.expleague.ml.data.tools.Pool;

/**
 * Created by noxoomo on 15/10/2017.
 */
public interface PoolReader {
  Pool read(final String path);
}

