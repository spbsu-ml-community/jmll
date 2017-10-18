package com.expleague.ml.cli.modes.impl;

import com.expleague.ml.cli.JMLLCLI;
import com.expleague.ml.cli.builders.data.DataBuilder;
import com.expleague.ml.cli.builders.data.impl.DataBuilderClassic;
import com.expleague.ml.cli.builders.data.impl.PoolReaderFeatureTxt;
import com.expleague.ml.cli.builders.data.impl.PoolReaderJson;
import com.expleague.ml.cli.modes.AbstractMode;
import com.expleague.ml.cli.modes.CliPoolReaderHelper;
import com.expleague.ml.data.tools.Pool;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class ValidatePool extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException {
    if (!command.hasOption(JMLLCLI.LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }

    final DataBuilder dataBuilder = new DataBuilderClassic();
    dataBuilder.setLearnPath(command.getOptionValue(JMLLCLI.LEARN_OPTION));
    dataBuilder.setReader(command.hasOption(JMLLCLI.JSON_FORMAT) ? new PoolReaderJson() : new PoolReaderFeatureTxt());
    try {
      CliPoolReaderHelper.setPoolReader(command, dataBuilder);
      final Pool pool = dataBuilder.create().getFirst();
      System.out.println("Valid pool");
    } catch (Exception e) {
      System.out.println("Invalid pool: can't even load");
    }
  }
}
