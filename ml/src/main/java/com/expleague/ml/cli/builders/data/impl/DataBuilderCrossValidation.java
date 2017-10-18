package com.expleague.ml.cli.builders.data.impl;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.cli.builders.data.DataBuilder;
import com.expleague.ml.cli.builders.data.PoolReader;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.data.tools.SubPool;
import com.expleague.commons.util.Pair;
import com.expleague.ml.data.tools.DataTools;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class DataBuilderCrossValidation implements DataBuilder {
  private String learnPath;
  private long randomSeed = System.currentTimeMillis();
  private double partition = 0.8;
  private PoolReader reader;

  @Override
  public void setLearnPath(final String learnPath) {
    this.learnPath = learnPath;
  }

  @Override
  public void setReader(final PoolReader reader) {
    this.reader = reader;
  }

  public void setRandomSeed(final long randomSeed) {
    this.randomSeed = randomSeed;
  }

  public void setPartition(final String partition) {
    try {
      this.partition = Double.parseDouble(partition);
    }
    catch (NumberFormatException e) {
      this.partition = 1. / Integer.parseInt(partition);
    }
  }

  @Override
  public Pair<Pool, Pool> create() {
    final Pool pool = reader.read(learnPath);
    final FastRandom rnd = new FastRandom(randomSeed);

    final int[][] cvSplit = DataTools.splitAtRandom(pool.size(), rnd, partition, 1.0 - partition);
    return Pair.create(new SubPool(pool, cvSplit[0]), new SubPool(pool, cvSplit[1]));
  }
}
