package com.expleague.ml.cli.modes.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.BFGrid;
import com.expleague.ml.FeatureExtractorsBuilder;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.cli.modes.CliPoolReaderHelper;
import com.expleague.ml.cli.output.ModelWriter;
import com.expleague.commons.func.WeakListenerHolder;
import com.expleague.commons.io.StreamTools;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.text.StringUtils;
import com.expleague.commons.util.Pair;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.cli.output.printers.*;
import com.expleague.ml.cli.builders.data.DataBuilder;
import com.expleague.ml.cli.builders.data.impl.DataBuilderClassic;
import com.expleague.ml.cli.builders.data.impl.DataBuilderCrossValidation;
import com.expleague.ml.cli.builders.methods.MethodsBuilder;
import com.expleague.ml.cli.builders.methods.grid.DynamicGridBuilder;
import com.expleague.ml.cli.modes.AbstractMode;
import com.expleague.ml.data.ctrs.CtrEstimationPolicy;
import com.expleague.ml.data.ctrs.CtrTarget;
import com.expleague.ml.data.tools.CatboostPool;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.loss.blockwise.BlockwiseMultiLabelLogit;
import com.expleague.ml.loss.multiclass.ClassicMulticlassLoss;
import com.expleague.ml.loss.multilabel.ClassicMultiLabelLoss;
import com.expleague.ml.loss.multilabel.MultiLabelOVRLogit;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.multiclass.JoinedBinClassModel;
import com.expleague.ml.cli.builders.methods.grid.GridBuilder;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.MissingArgumentException;

import java.io.File;

import static com.expleague.ml.cli.JMLLCLI.*;

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
    CliPoolReaderHelper.setPoolReader(command, dataBuilder);

    final Pair<? extends Pool, ? extends Pool> pools = dataBuilder.create();
    final Pool learn = pools.getFirst();
    final Pool test = pools.getSecond();


    final FeatureExtractorsBuilder featureExtractorsBuilder = new FeatureExtractorsBuilder(learn);
    //loading grid (if needed)
    final GridBuilder gridBuilder = new GridBuilder();
    if (command.hasOption(GRID_OPTION)) {
      gridBuilder.setGrid(BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION)))));
    } else {
      gridBuilder.setBinsCount(Integer.valueOf(command.getOptionValue(BIN_FOLDS_COUNT_OPTION, "32")));
      gridBuilder.setDataSet(learn.vecData());
      if (learn instanceof CatboostPool) {
        gridBuilder.addCatFeatureIds(((CatboostPool) learn).catFeatureIds());
      }
    }

    final DynamicGridBuilder dynamicGridBuilder = new DynamicGridBuilder();
    dynamicGridBuilder.setBinsCount(Integer.valueOf(command.getOptionValue(BIN_FOLDS_COUNT_OPTION, "1")));
    dynamicGridBuilder.setDataSet(learn.vecData());


    //choose optimization method
    final MethodsBuilder methodsBuilder = new MethodsBuilder();
    final FastRandom random = new FastRandom();
    methodsBuilder.setRandom(random);
    methodsBuilder.setGridBuilder(gridBuilder);
    methodsBuilder.setDynamicGridBuilder(dynamicGridBuilder);
    methodsBuilder.setFeaturesExtractorBuilder(featureExtractorsBuilder);
    final VecOptimization method = methodsBuilder.create(command.getOptionValue(OPTIMIZATION_OPTION, DEFAULT_OPTIMIZATION_SCHEME));

    //set target
    final String target = command.getOptionValue(TARGET_OPTION, DEFAULT_TARGET);
    final TargetFunc loss = learn.target(DataTools.targetByName(target));

    final CtrEstimationPolicy ctrEstimationPolicy = CtrEstimationPolicy.valueOf(command.getOptionValue(CTRS_ESTIMATION, CtrEstimationPolicy.Greedy.name()));
    featureExtractorsBuilder.setEstimationPolicy(ctrEstimationPolicy);
    featureExtractorsBuilder.useRandomPermutation(random);
    final String[] ctrs = command.getOptionValues(CTRS_OPTION);
    if (ctrs != null) {
      for (final String ctrName : ctrs) {
        featureExtractorsBuilder.addCtrs(new CtrTarget((Vec) learn.target(0), CtrTarget.CtrTargetType.valueOf(ctrName)));
      }
    }
    if (command.hasOption(ONE_HOT_LIMIT)) {
      featureExtractorsBuilder.useOneHots(Integer.valueOf(command.getOptionValue(ONE_HOT_LIMIT)));
    }

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
