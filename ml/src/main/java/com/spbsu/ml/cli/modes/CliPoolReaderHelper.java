package com.spbsu.ml.cli.modes;

import com.spbsu.ml.cli.builders.data.DataBuilder;
import org.apache.commons.cli.CommandLine;

import java.io.IOException;

import static com.spbsu.ml.cli.JMLLCLI.*;
import static com.spbsu.ml.cli.builders.data.ReaderFactory.createCatBoostPoolReader;
import static com.spbsu.ml.cli.builders.data.ReaderFactory.createFeatureTxtReader;
import static com.spbsu.ml.cli.builders.data.ReaderFactory.createJsonReader;

/**
 * Created by noxoomo on 15/10/2017.
 */
public class CliPoolReaderHelper {

  public static DataBuilder setPoolReader(final CommandLine command,
                                          final DataBuilder builder) throws IOException {
    if (command.hasOption(CD_FILE)) {
      builder.setReader(createCatBoostPoolReader(command.getOptionValue(CD_FILE),
          command.getOptionValue(LEARN_OPTION),
          toCharChecked(command.getOptionValue(DELIMITER, "\t")),
          command.hasOption(HAS_HEADER)));
    }
    else {
      final boolean isJson = command.hasOption(JSON_FORMAT);
      builder.setReader(isJson ? createJsonReader() : createFeatureTxtReader());
    }
    return builder;
  }

  private static char toCharChecked(final String optionValue) {
    if (optionValue.length() != 1) {
      throw new RuntimeException("Error: wrong option format for char value " + optionValue);
    }
    return optionValue.charAt(0);
  }
}
