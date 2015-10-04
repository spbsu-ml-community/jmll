package com.spbsu.ml.cli.modes;

import org.apache.commons.cli.CommandLine;

import static com.spbsu.ml.cli.JMLLCLI.LEARN_OPTION;
import static com.spbsu.ml.cli.JMLLCLI.OUTPUT_OPTION;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public abstract class AbstractMode {

  public abstract void run(CommandLine command) throws Exception;

  protected static String getOutputName(final CommandLine command) {
    final String outName;
    if (command.hasOption(OUTPUT_OPTION)) {
      outName = command.getOptionValue(OUTPUT_OPTION);
    } else {
      final String tempName = command.getOptionValue(LEARN_OPTION, "features.txt");
      if (tempName.endsWith(".gz"))
        outName = tempName.substring(0, tempName.length() - ".gz".length());
      else
        outName = tempName;
    }
    return outName;
  }
}
