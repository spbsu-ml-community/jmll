package com.expleague.ml.cli.modes;

import com.expleague.ml.cli.builders.data.DataBuilder;
import org.apache.commons.cli.CommandLine;

import java.io.IOException;

import static com.expleague.ml.cli.JMLLCLI.*;
import static com.expleague.ml.cli.builders.data.ReaderFactory.createCatBoostPoolReader;
import static com.expleague.ml.cli.builders.data.ReaderFactory.createFeatureTxtReader;
import static com.expleague.ml.cli.builders.data.ReaderFactory.createJsonReader;
import static com.expleague.ml.cli.builders.data.ReaderFactory.createLetorReader;

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
    else if (command.hasOption(JSON_FORMAT)) {
      builder.setReader(createJsonReader());
    } else if (command.hasOption(LETOR_FORMAT)) {
      builder.setReader(createLetorReader());
    } else {
      builder.setReader(createFeatureTxtReader());
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
