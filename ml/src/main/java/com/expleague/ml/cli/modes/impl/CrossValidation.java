package com.expleague.ml.cli.modes.impl;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.util.BestHolder;
import com.expleague.ml.cli.builders.data.impl.DataBuilderClassic;
import com.expleague.ml.cli.cv.KFoldCrossValidation;
import com.expleague.ml.cli.gridsearch.ParametersExtractor;
import com.expleague.commons.io.StreamTools;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.text.StringUtils;
import com.expleague.ml.cli.builders.methods.MethodsBuilder;
import com.expleague.ml.cli.builders.methods.grid.GridBuilder;
import com.expleague.ml.cli.modes.AbstractMode;
import com.expleague.ml.cli.modes.CliPoolReaderHelper;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.BFGrid;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.expleague.ml.cli.JMLLCLI.*;

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
  private static final Logger LOG = LoggerFactory.getLogger(CrossValidation.class);

  public void run(final CommandLine command) throws MissingArgumentException, IOException {

    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }

    final boolean schemeBatchComparision = command.hasOption(LOAD_OPTIMIZATION_SCHEMES_FROM_FILE_OPTION);

    if (!schemeBatchComparision && !command.hasOption(OPTIMIZATION_OPTION)) {
      throw new MissingArgumentException("Please provide 'OPTIMIZATION_OPTION'");
    }
    if (!command.hasOption(CROSS_VALIDATION_OPTION)) {
      throw new MissingArgumentException("Please provide 'CROSS_VALIDATION_OPTIONS");
    }

    final DataBuilderClassic dataBuilder;
    dataBuilder = new DataBuilderClassic();
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    CliPoolReaderHelper.setPoolReader(command, dataBuilder);
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
    final KFoldCrossValidation crossValidation = new KFoldCrossValidation(sourcePool, random, foldsCount, targetClassName, methodsBuilder);

    if (schemeBatchComparision) {
      if (command.hasOption(RANGES_OPTION)) {
        throw new RuntimeException("Error: range option is not supported for batch model comparision cv mode");
      }
      List<String> schemes = loadSchemesFromFile(command.getOptionValue(LOAD_OPTIMIZATION_SCHEMES_FROM_FILE_OPTION));
      KFoldCrossValidation.CrossValidationModelComparisonResult result = crossValidation.evaluateSchemesBatch(schemes);
      dumpResult(result, command.getOptionValue(CROSS_VALIDATION_RESULT_OPTION, DEFAULT_MODELS_COMPARISION_CV_OUTPUT_FILE));
    } else {
      final String commonScheme = command.getOptionValue(OPTIMIZATION_OPTION);

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

  private void dumpResult(KFoldCrossValidation.CrossValidationModelComparisonResult result, final String fileName) throws IOException {
    LOG.debug("Schemes");
    final BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
    final String scheme = result.getSchemes().collect(Collectors.joining("\t"));
    writer.write("Scheme\n");
    writer.write(scheme  + "\n");
    LOG.debug(scheme);

    final String modelScores = Arrays.stream(result.getScores().toArray()).mapToObj(String::valueOf).collect(Collectors.joining("\t"));

    LOG.debug("Scores for models");
    LOG.debug(modelScores);

    writer.write("scores\n");
    writer.write(modelScores + "\n");

    LOG.debug("Pairwise scores diffs");
    final String tsvDiffs = matrixToTsv(result.getPairwiseDiffs());
    LOG.debug(tsvDiffs);
    writer.write("scores_diffs\n");
    writer.write(tsvDiffs);

    LOG.debug("WX test");
    final String wxTable = matrixToTsv(result.getWxStats());
    LOG.debug(wxTable);
    writer.write("wx_test_table\n");
    writer.write(wxTable);
    writer.flush();
    writer.close();
  }

  private String matrixToTsv(Mx data) {
    final StringBuilder builder = new StringBuilder();
    for (int i = 0; i < data.rows(); ++i) {
      for (int j = 0; j < data.columns(); ++j) {
        builder.append(data.get(i, j));
        if (j + 1 != data.columns()) {
          builder.append("\t");
        }
      }
      builder.append("\n");
    }
    return builder.toString();
  }

  private List<String> loadSchemesFromFile(String file) {
    try {
      final BufferedReader reader = new BufferedReader(new FileReader(file));
      return reader.lines().collect(Collectors.toList());
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
}
