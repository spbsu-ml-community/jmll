package com.spbsu.ml.cli.modes.impl;

import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderClassic;
import com.spbsu.ml.cli.builders.methods.grid.GridBuilder;
import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.meta.DSItem;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;
import java.io.IOException;

import static com.spbsu.ml.cli.JMLLCLI.*;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class EvaluateModel extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(MODEL_OPTION))
      throw new MissingArgumentException("Please provide 'MODEL_OPTION'");
    if (!command.hasOption(METRICS_OPTION))
      throw new MissingArgumentException("Please provide 'METRICS_OPTION'");
    if (!command.hasOption(TEST_OPTION))
      throw new MissingArgumentException("Please provide 'TEST_OPTION'");

    final Trans model;
    { // loading model
      final ModelsSerializationRepository serializationRepository;
      if (command.hasOption(GRID_OPTION)) {
        final GridBuilder gridBuilder = new GridBuilder();
        gridBuilder.setGrid(BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION)))));
        serializationRepository = new ModelsSerializationRepository(gridBuilder.create());
      }
      else {
        serializationRepository = new ModelsSerializationRepository();
      }
      try {
        model = DataTools.readModel(command.getOptionValue(MODEL_OPTION), serializationRepository);
      }
      catch (ClassNotFoundException e) {
        throw new RuntimeException(e);
      }
    }
    final Pool<? extends DSItem> pool;
    { // loading data
      final DataBuilderClassic dataBuilder = new DataBuilderClassic();
      dataBuilder.setLearnPath(command.getOptionValue(TEST_OPTION));
      dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));
      //noinspection unchecked
      pool = dataBuilder.create().getFirst();
    }
    {
      final Mx mx = model.transAll(pool.vecData().data());
      for (final String metricName : command.getOptionValues(METRICS_OPTION)) {
        final TargetFunc target = pool.target(DataTools.targetByName(metricName));
        target.printResult(mx, System.out);
      }
    }
  }
}
