package com.spbsu.ml.cli.modes.impl;

import com.spbsu.ml.cli.builders.data.DataBuilder;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderClassic;
import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import static com.spbsu.ml.cli.JMLLCLI.JSON_FORMAT;
import static com.spbsu.ml.cli.JMLLCLI.LEARN_OPTION;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class PrintPoolInfo extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide: learn_option");
    }

    final DataBuilder builder = new DataBuilderClassic();
    builder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    builder.setJsonFormat(command.hasOption(JSON_FORMAT));
    final Pool<?> pool = builder.create().getFirst();
    System.out.println(DataTools.getPoolInfo(pool));
  }
}
