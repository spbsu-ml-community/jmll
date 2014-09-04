package com.spbsu.ml.cli.builders.data.impl;

import com.spbsu.commons.util.Pair;
import com.spbsu.ml.cli.builders.data.DataBuilder;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;

import java.io.IOException;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class DataBuilderClassic implements DataBuilder {
  private String learnPath;
  private String testPath;
  private boolean isJsonFormat = false;

  @Override
  public void setLearnPath(final String learnPath) {
    this.learnPath = learnPath;
  }

  public void setTestPath(final String testPath) {
    this.testPath = testPath;
  }

  @Override
  public void setJsonFormat(final boolean isJsonFormat) {
    this.isJsonFormat = isJsonFormat;
  }

  @Override
  public Pair<Pool, Pool> create() {
    try {
      final Pool learn = isJsonFormat ? DataTools.loadFromFile(learnPath) : DataTools.loadFromFeaturesTxt(learnPath);
      final Pool test;
      if (testPath != null && !testPath.equals(learnPath)) {
        test = isJsonFormat ? DataTools.loadFromFile(testPath) : DataTools.loadFromFeaturesTxt(testPath);
      } else {
        test = learn;
      }
      return Pair.create(learn, test);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
