package com.expleague.ml.cli.modes.impl;

import com.expleague.commons.io.StreamTools;
import com.expleague.commons.util.Pair;
import com.expleague.ml.BFGrid;
import com.expleague.ml.cli.builders.methods.grid.GridBuilder;
import com.expleague.ml.cli.modes.AbstractMode;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.io.ModelsSerializationRepository;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;
import java.io.IOException;

import static com.expleague.ml.cli.JMLLCLI.GRID_OPTION;
import static com.expleague.ml.cli.JMLLCLI.MODEL_OPTION;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class ValidateModel extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(MODEL_OPTION)) {
      throw new MissingArgumentException("Please provide 'MODEL_OPTION'");
    }

    final ModelsSerializationRepository serializationRepository;
    if (command.hasOption(GRID_OPTION)) {
      final GridBuilder gridBuilder = new GridBuilder();
      gridBuilder.setGrid(BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION)))));
      serializationRepository = new ModelsSerializationRepository(gridBuilder.create());
    } else {
      serializationRepository = new ModelsSerializationRepository();
    }

    final Pair<Boolean, String> validationResults = DataTools.validateModel(command.getOptionValue(MODEL_OPTION), serializationRepository);
    System.out.println(validationResults.getSecond());
  }
}
