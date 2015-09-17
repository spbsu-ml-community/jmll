package com.spbsu.ml.cli.modes.impl;

import com.spbsu.commons.text.StringUtils;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderCrossValidation;
import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.FileWriter;
import java.io.IOException;

import static com.spbsu.ml.cli.JMLLCLI.*;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class SplitJsonPool extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION) && !command.hasOption(JSON_FORMAT) && !command.hasOption(CROSS_VALIDATION_OPTION)) {
      throw new MissingArgumentException("Please provide: learn_option, json_flag and cross_validation_option");
    }

    final DataBuilderCrossValidation builder = new DataBuilderCrossValidation();
    builder.setJsonFormat(command.hasOption(JSON_FORMAT));
    builder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    final String[] cvOptions = StringUtils.split(command.getOptionValue(CROSS_VALIDATION_OPTION), "/", 2);
    builder.setRandomSeed(Integer.valueOf(cvOptions[0]));
    builder.setPartition(cvOptions[1]);

    final Pair<? extends Pool, ? extends Pool> pools = builder.create();

    final String outputName = getOutputName(command);
    DataTools.writePoolTo(pools.getFirst(), new FileWriter(outputName + ".learn"));
    DataTools.writePoolTo(pools.getSecond(), new FileWriter(outputName + ".test"));
  }
}
