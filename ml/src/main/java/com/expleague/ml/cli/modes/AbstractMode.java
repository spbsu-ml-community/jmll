package com.expleague.ml.cli.modes;

import com.expleague.ml.cli.JMLLCLI;
import org.apache.commons.cli.CommandLine;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public abstract class AbstractMode {

  public abstract void run(CommandLine command) throws Exception;

  protected static String getOutputName(final CommandLine command) {
    final String outName;
    if (command.hasOption(JMLLCLI.OUTPUT_OPTION)) {
      outName = command.getOptionValue(JMLLCLI.OUTPUT_OPTION);
    } else {
      final String tempName = command.getOptionValue(JMLLCLI.LEARN_OPTION, "features.txt");
      if (tempName.endsWith(".gz"))
        outName = tempName.substring(0, tempName.length() - ".gz".length());
      else
        outName = tempName;
    }
    return outName;
  }
}
