package com.expleague.ml.cli.modes.impl;

import com.expleague.ml.cli.JMLLCLI;
import com.expleague.ml.cli.builders.data.impl.DataBuilderClassic;
import com.expleague.ml.cli.modes.AbstractMode;
import com.expleague.ml.cli.modes.CliPoolReaderHelper;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class ConvertPoolLibSvm extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(JMLLCLI.LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION");
    }

    final DataBuilderClassic dataBuilder = new DataBuilderClassic();
    CliPoolReaderHelper.setPoolReader(command, dataBuilder);
    dataBuilder.setLearnPath(command.getOptionValue(JMLLCLI.LEARN_OPTION));
    final Pool<?> pool = dataBuilder.create().getFirst();
    final String outputName = command.hasOption(JMLLCLI.OUTPUT_OPTION) ? getOutputName(command) : getOutputName(command) + ".libfm";
    try (final BufferedWriter out = new BufferedWriter(new FileWriter(outputName))) {
      DataTools.writePoolInLibfmFormat(pool, out);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
