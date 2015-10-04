package com.spbsu.ml.cli.modes.impl;

import com.spbsu.ml.cli.builders.data.DataBuilder;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderClassic;
import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.IOException;

import static com.spbsu.ml.cli.JMLLCLI.*;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class ConvertPoolJson2Classsic extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }

    final DataBuilder dataBuilder = new DataBuilderClassic();
    dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    final Pool pool = dataBuilder.create().getFirst();
    final String outputName = command.hasOption(OUTPUT_OPTION) ? getOutputName(command) : getOutputName(command) + ".tsv";
    DataTools.writeClassicPoolTo(pool, outputName);
  }
}
