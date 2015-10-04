package com.spbsu.ml.cli.modes.impl;

import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.text.StringUtils;
import com.spbsu.commons.util.BestHolder;
import com.spbsu.commons.util.logging.Logger;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderClassic;
import com.spbsu.ml.cli.builders.methods.MethodsBuilder;
import com.spbsu.ml.cli.builders.methods.grid.GridBuilder;
import com.spbsu.ml.cli.cv.KFoldCrossValidation;
import com.spbsu.ml.cli.gridsearch.ParametersExtractor;
import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.data.tools.Pool;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import static com.spbsu.ml.cli.JMLLCLI.*;

/**
 * User: qdeee
 * Date: 16.09.15
 *
 * Запуск K-fold кросс-валидации: делим пул на K фолдов, на K-1 обучаемся, на последнем измеряем целевую функцию, повторяем K  раз.
 * Далее усредняем по числу фолдов.
 *
 * Количество фолдов и сид рандома передаются через CROSS_VALIDATION_OPTIONS в формате -X <seed>/<folds_count> (-X 100500/3)
 * Доступно 2 варианта запуска:
 *    1. Перебор сетки параметров.
 *        В команде запуска перебираемые параметры оставляем как %s и добавляем -r <start>:<end>:<step>. Пример команды :
 *        "cross-validation -X 100500/3 -f pool.tsv  -T blockwise.BlockwiseMLLLogit -O GradientBoosting(weak=MultiClassSplit(weak=GreedyObliviousTree),iterations=%s,step=%s) -r 10:15:1;0.1:0.9:0.1"
 *    2. Одиночный запуск для фиксированных значений параметров. В этом случае передавать RANGES не нужно, а параметры указываются как обычно.
 */
public class CrossValidation extends AbstractMode {
  private static final Logger LOG = Logger.create(CrossValidation.class);

  public void run(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }
    if (!command.hasOption(OPTIMIZATION_OPTION)) {
      throw new MissingArgumentException("Please provide 'OPTIMIZATION_OPTION'");
    }
    if (!command.hasOption(CROSS_VALIDATION_OPTION)) {
      throw new MissingArgumentException("Please provide 'CROSS_VALIDATION_OPTIONS");
    }

    final DataBuilderClassic dataBuilder;
    dataBuilder = new DataBuilderClassic();
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));
    final Pool sourcePool = dataBuilder.create().getFirst();

    final GridBuilder gridBuilder = new GridBuilder();
    if (command.hasOption(GRID_OPTION)) {
      gridBuilder.setGrid(BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION)))));
    } else {
      gridBuilder.setBinsCount(Integer.valueOf(command.getOptionValue(BIN_FOLDS_COUNT_OPTION, "32")));
      gridBuilder.setDataSet(sourcePool.vecData());
    }

    final String[] cvOptions = StringUtils.split(command.getOptionValue(CROSS_VALIDATION_OPTION), "/", 2);
    final FastRandom random = new FastRandom(Long.valueOf(cvOptions[0]));
    final int foldsCount = Integer.parseInt(cvOptions[1]);

    final MethodsBuilder methodsBuilder = new MethodsBuilder();
    methodsBuilder.setRandom(random);
    methodsBuilder.setGridBuilder(gridBuilder);


    final String targetClassName = command.getOptionValue(TARGET_OPTION, DEFAULT_TARGET);
    final String commonScheme = command.getOptionValue(OPTIMIZATION_OPTION);

    final KFoldCrossValidation crossValidation = new KFoldCrossValidation(sourcePool, random, foldsCount, targetClassName, methodsBuilder);
    final double score;
    if (command.hasOption(RANGES_OPTION)) {
      final String[][] parametersSpace = ParametersExtractor.parse(command.getOptionValue(RANGES_OPTION));
      final BestHolder<Object[]> bestHolder = crossValidation.evaluateParametersRange(commonScheme, parametersSpace);
      score = bestHolder.getScore();
      LOG.info("Best parameters: " + Arrays.toString(bestHolder.getValue()));

    } else {
      score = crossValidation.evaluate(commonScheme);
    }

    LOG.info("Final score: " + score);
  }
}
