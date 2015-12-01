package com.spbsu.ml.cli.modes.impl;

import com.spbsu.commons.func.WeakListenerHolder;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.text.StringUtils;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.logging.Interval;
import com.spbsu.ml.*;
import com.spbsu.ml.cli.builders.data.DataBuilder;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderClassic;
import com.spbsu.ml.cli.builders.data.impl.DataBuilderCrossValidation;
import com.spbsu.ml.cli.builders.methods.MethodsBuilder;
import com.spbsu.ml.cli.builders.methods.grid.DynamicGridBuilder;
import com.spbsu.ml.cli.builders.methods.grid.GridBuilder;
import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.cli.output.ModelWriter;
import com.spbsu.ml.cli.output.printers.*;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseMultiLabelLogit;
import com.spbsu.ml.loss.multiclass.ClassicMulticlassLoss;
import com.spbsu.ml.loss.multilabel.ClassicMultiLabelLoss;
import com.spbsu.ml.loss.multilabel.MultiLabelOVRLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.JoinedBinClassModel;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;

import static com.spbsu.ml.cli.JMLLCLI.*;

/**
 * User: qdeee
 * Date: 16.09.15
 */
public class Fit extends AbstractMode {

  @Override
  public void run(final CommandLine command) throws Exception {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }

    //data loading
    final DataBuilder dataBuilder;
    if (command.hasOption(CROSS_VALIDATION_OPTION)) {
      final DataBuilderCrossValidation dataBuilderCrossValidation = new DataBuilderCrossValidation();
      final String[] cvOptions = StringUtils.split(command.getOptionValue(CROSS_VALIDATION_OPTION), "/", 2);
      dataBuilderCrossValidation.setRandomSeed(Long.valueOf(cvOptions[0]));
      dataBuilderCrossValidation.setPartition(cvOptions[1]);
      dataBuilder = dataBuilderCrossValidation;
    } else {
      dataBuilder = new DataBuilderClassic();
      ((DataBuilderClassic) dataBuilder).setTestPath(command.getOptionValue(TEST_OPTION));
    }
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));

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

    final DynamicGridBuilder dynamicGridBuilder = new DynamicGridBuilder();
    dynamicGridBuilder.setBinsCount(Integer.valueOf(command.getOptionValue(BIN_FOLDS_COUNT_OPTION, "1")));
    dynamicGridBuilder.setDataSet(learn.vecData());


    //choose optimization method
    final MethodsBuilder methodsBuilder = new MethodsBuilder();
    methodsBuilder.setRandom(new FastRandom());
    methodsBuilder.setGridBuilder(gridBuilder);
    methodsBuilder.setDynamicGridBuilder(dynamicGridBuilder);
    final VecOptimization method = methodsBuilder.create(command.getOptionValue(OPTIMIZATION_OPTION, DEFAULT_OPTIMIZATION_SCHEME));

    //set target
    final String target = command.getOptionValue(TARGET_OPTION, DEFAULT_TARGET);
    final TargetFunc loss = learn.target(DataTools.targetByName(target));

    //set metrics
    final String[] metricNames = command.getOptionValues(METRICS_OPTION);
    final Func[] metrics;
    if (metricNames != null) {
      metrics = new Func[metricNames.length];
      for (int i = 0; i < metricNames.length; i++) {
        metrics[i] = test.target(DataTools.targetByName(metricNames[i]));
      }
    } else {
      metrics = new Func[]{test.target(DataTools.targetByName(target))};
    }


    //added progress handlers
    ProgressHandler progressPrinter = null;
    if (method instanceof WeakListenerHolder && command.hasOption(VERBOSE_OPTION) && !command.hasOption(FAST_OPTION)) {
      final int printPeriod = Integer.valueOf(command.getOptionValue(PRINT_PERIOD, "10"));
      if (loss instanceof BlockwiseMLLLogit) {
        progressPrinter = new MulticlassProgressPrinter(learn, test, printPeriod); //f*ck you with your custom different-dimensional metrics
      } else if (loss instanceof BlockwiseMultiLabelLogit || loss instanceof MultiLabelOVRLogit) {
        progressPrinter = new MultiLabelLogitProgressPrinter(learn, test, printPeriod);
      } else {
        progressPrinter = new DefaultProgressPrinter(learn, test, loss, metrics, printPeriod);
      }
      //noinspection unchecked
      ((WeakListenerHolder) method).addListener(progressPrinter);
    }

    if (method instanceof WeakListenerHolder && command.hasOption(HIST_OPTION)) {
      final ProgressHandler histogramPrinter = new HistogramPrinter();
      //noinspection unchecked
      ((WeakListenerHolder) method).addListener(histogramPrinter);
    }


    //fitting
    Interval.start();
    Interval.suspend();
    final Trans result = method.fit(learn.vecData(), loss);
    Interval.stopAndPrint("Total fit time:");


    //calc & print scores
    if (!command.hasOption(FAST_OPTION) && !command.hasOption(SKIP_FINAL_EVAL_OPTION)) {
      ResultsPrinter.printResults(result, learn, test, loss, metrics);
      if (loss instanceof BlockwiseMLLLogit) {
        ResultsPrinter.printMulticlassResults(result, learn, test);
      } else if (loss instanceof ClassicMulticlassLoss) {
        final int printPeriod = Integer.valueOf(command.getOptionValue(PRINT_PERIOD, "20"));
        MCTools.makeOneVsRestReport(learn, test, (JoinedBinClassModel) result, printPeriod);
      } else if (loss instanceof ClassicMultiLabelLoss || loss instanceof BlockwiseMultiLabelLogit || loss instanceof MultiLabelOVRLogit) {
        ResultsPrinter.printMultilabelResult(result, learn, test);
      }
    }


    //write model
    final String outName = getOutputName(command);
    final ModelWriter modelWriter = new ModelWriter(outName);

    if (command.hasOption(WRITE_BIN_FORMULA)) {
      modelWriter.tryWriteBinFormula(result);
    } else {
      if (!command.hasOption(GRID_OPTION)) {
        modelWriter.tryWriteGrid(result);
      }
      modelWriter.tryWriteDynamicGrid(result);
      modelWriter.writeModel(result);
    }
  }

}
