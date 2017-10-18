package com.expleague.ml.cli.modes.impl;

import com.expleague.ml.cli.gridsearch.ParametersExtractor;
import com.expleague.commons.io.StreamTools;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.text.StringUtils;
import com.expleague.commons.util.Pair;
import com.expleague.ml.BFGrid;
import com.expleague.commons.math.Func;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.cli.builders.data.DataBuilder;
import com.expleague.ml.cli.builders.data.impl.DataBuilderClassic;
import com.expleague.ml.cli.builders.data.impl.DataBuilderCrossValidation;
import com.expleague.ml.cli.builders.methods.MethodsBuilder;
import com.expleague.ml.cli.builders.methods.grid.GridBuilder;
import com.expleague.ml.cli.gridsearch.OptimumHolder;
import com.expleague.ml.cli.modes.AbstractMode;
import com.expleague.ml.cli.modes.CliPoolReaderHelper;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;
import java.io.IOException;

import static com.expleague.ml.cli.JMLLCLI.*;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class GridSearch extends AbstractMode {

  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }
    if (!command.hasOption(OPTIMIZATION_OPTION)) {
      throw new MissingArgumentException("Please provide 'OPTIMIZATION_OPTION'");
    }
    if (!command.hasOption(RANGES_OPTION)) {
      throw new MissingArgumentException("Please provide 'RANGE_OPTION'");
    }

    //data loading
    final DataBuilder dataBuilder;
    if (command.hasOption(CROSS_VALIDATION_OPTION)) {
      final DataBuilderCrossValidation dataBuilderCrossValidation = new DataBuilderCrossValidation();
      final String[] cvOptions = StringUtils.split(command.getOptionValue(CROSS_VALIDATION_OPTION), "/", 2);
      dataBuilderCrossValidation.setRandomSeed(Integer.valueOf(cvOptions[0]));
      dataBuilderCrossValidation.setPartition(cvOptions[1]);
      dataBuilder = dataBuilderCrossValidation;
    } else {
      dataBuilder = new DataBuilderClassic();
      ((DataBuilderClassic) dataBuilder).setTestPath(command.getOptionValue(TEST_OPTION));
    }
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    CliPoolReaderHelper.setPoolReader(command, dataBuilder);


    final Pair<? extends Pool, ? extends Pool> pools = dataBuilder.create();
    final Pool learn = pools.getFirst();
    final Pool test = pools.getSecond();

    //loading grid (if needed)
    final GridBuilder gridBuilder = new GridBuilder();
    if (command.hasOption(GRID_OPTION)) {
      gridBuilder.setGrid(BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION)))));
    } else {
      gridBuilder.setBinsCount(Integer.valueOf(command.getOptionValue(BIN_FOLDS_COUNT_OPTION, "32")));
      gridBuilder.setDataSet(learn.vecData());
    }

    //choose optimization method
    final MethodsBuilder methodsBuilder = new MethodsBuilder();
    methodsBuilder.setRandom(new FastRandom());
    methodsBuilder.setGridBuilder(gridBuilder);

    //set target
    final String target = command.getOptionValue(TARGET_OPTION, DEFAULT_TARGET);
    final TargetFunc loss = learn.target(DataTools.targetByName(target));

    //set metrics
    final String[] metricNames = command.getOptionValues(METRICS_OPTION);
    final Func[] metrics;
    if (metricNames != null) {
      metrics = new Func[metricNames.length];
      for (int i = 0; i < metricNames.length; i++) {
        metrics[i] = test.targetByName(metricNames[i]);
      }
    } else {
      metrics = new Func[]{test.targetByName(target)};
    }

    final com.expleague.ml.cli.gridsearch.GridSearch gridSearch = new com.expleague.ml.cli.gridsearch.GridSearch(learn, test, loss, metrics, methodsBuilder);
    final String commonScheme = command.getOptionValue(OPTIMIZATION_OPTION);
    final String[][] parametersSpace = ParametersExtractor.parse(command.getOptionValue(RANGES_OPTION));
    final OptimumHolder[] searchResult = gridSearch.search(commonScheme, parametersSpace);
    for (int i = 0; i < metrics.length; i++) {
      System.out.println(metrics[i].getClass().getSimpleName() + " : " + searchResult[i]);
    }
  }
}
