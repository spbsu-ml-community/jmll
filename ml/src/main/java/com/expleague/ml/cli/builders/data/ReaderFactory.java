package com.expleague.ml.cli.builders.data;


import com.expleague.ml.cli.builders.data.impl.CatBoostPoolReader;
import com.expleague.ml.cli.builders.data.impl.PoolReaderFeatureTxt;
import com.expleague.ml.cli.builders.data.impl.PoolReaderLetor;
import com.expleague.ml.data.tools.CatBoostPoolDescription;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.cli.builders.data.impl.PoolReaderJson;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by noxoomo on 15/10/2017.
 */
public class ReaderFactory {
  public static PoolReader createJsonReader() {
    return new PoolReaderJson();
  }

  public static PoolReader createFeatureTxtReader() {
    return new PoolReaderFeatureTxt();
  }

  public static PoolReader createLetorReader() {
    return new PoolReaderLetor();
  }

  public static PoolReader createCatBoostPoolReader(final String cdFile,
                                                    final String firstLineFileProvider,
                                                    final char sep,
                                                    final boolean hasHeader) throws IOException {
    final int columnCount = DataTools.getLineCount(DataTools.gzipOrFileReader(new File(firstLineFileProvider)), sep);
    final CatBoostPoolDescription description = new CatBoostPoolDescription.DescriptionBuilder(columnCount)
        .setDelimiter(sep)
        .setHasHeaderColumnFlag(hasHeader)
        .loadColumnDescription(new FileReader(cdFile))
        .description();
    return new CatBoostPoolReader(description);
  }
}
