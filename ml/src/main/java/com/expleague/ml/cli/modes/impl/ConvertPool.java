package com.expleague.ml.cli.modes.impl;

import com.expleague.ml.cli.builders.data.DataBuilder;
import com.expleague.ml.cli.builders.data.impl.DataBuilderClassic;
import com.expleague.ml.cli.modes.CliPoolReaderHelper;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.cli.modes.AbstractMode;
import com.expleague.ml.data.tools.DataTools;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.FileWriter;
import java.io.IOException;

import static com.expleague.ml.cli.JMLLCLI.*;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class ConvertPool extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }

    final DataBuilder dataBuilder = new DataBuilderClassic();
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    CliPoolReaderHelper.setPoolReader(command, dataBuilder);
    final Pool pool = dataBuilder.create().getFirst();
    final String outputName = command.hasOption(OUTPUT_OPTION) ? getOutputName(command) : getOutputName(command) + ".pool";
    DataTools.writePoolTo(pool, new FileWriter(outputName));
  }
}
