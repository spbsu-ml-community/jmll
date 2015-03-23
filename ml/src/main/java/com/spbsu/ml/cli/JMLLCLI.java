package com.spbsu.ml.cli;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.WeakListenerHolder;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqBuilder;
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
import com.spbsu.ml.cli.output.ModelWriter;
import com.spbsu.ml.cli.output.printers.DefaultProgressPrinter;
import com.spbsu.ml.cli.output.printers.HistogramPrinter;
import com.spbsu.ml.cli.output.printers.MulticlassProgressPrinter;
import com.spbsu.ml.cli.output.printers.ResultsPrinter;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multiclass.ClassicMulticlassLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.JoinedBinClassModel;
import org.apache.commons.cli.*;

import java.io.*;

/**
 * User: qdeee
 * Date: 03.09.14
 */
@SuppressWarnings("UnusedDeclaration,AccessStaticViaInstance")
public class JMLLCLI {
  public static final String DEFAULT_TARGET = "L2";
  public static final String DEFAULT_OPTIMIZATION_SCHEME = "GradientBoosting(local=SatL2, weak=GreedyObliviousTree(depth=6), step=0.02, iterations=1000)";

  private static final String LEARN_OPTION = "f";
  private static final String JSON_FORMAT = "j";
  private static final String TEST_OPTION = "t";
  private static final String CROSS_VALIDATION_OPTION = "X";

  private static final String TARGET_OPTION = "T";
  private static final String METRICS_OPTION = "M";

  private static final String BIN_FOLDS_COUNT_OPTION = "x";
  private static final String GRID_OPTION = "g";
  private static final String OPTIMIZATION_OPTION = "O";

  private static final String VERBOSE_OPTION = "v";
  private static final String PRINT_PERIOD = "printperiod";
  private static final String FAST_OPTION = "fast";
  private static final String HIST_OPTION = "h";
  private static final String OUTPUT_OPTION = "o";
  private static final String WRITE_BIN_FORMULA = "mxbin";

  private static final String MODEL_OPTION = "m";

  static Options options = new Options();

  static {
    options.addOption(OptionBuilder.withLongOpt("learn").withDescription("features.txt format file used as learn").hasArg().create(LEARN_OPTION));
    options.addOption(OptionBuilder.withLongOpt("json-format").withDescription("alternative format for features.txt").hasArg(false).create(JSON_FORMAT));
    options.addOption(OptionBuilder.withLongOpt("test").withDescription("test part in features.txt format").hasArg().create(TEST_OPTION));
    options.addOption(OptionBuilder.withLongOpt("cross-validation").withDescription("k folds CV").hasArg().create(CROSS_VALIDATION_OPTION));

    options.addOption(OptionBuilder.withLongOpt("target").withDescription("target function to optimize format Global/Weak/Cursor (" + DEFAULT_TARGET + ")").hasArg().create(TARGET_OPTION));
    options.addOption(OptionBuilder.withLongOpt("metrics").withDescription("metrics to test, by default contains global optimization target").hasArgs().create(METRICS_OPTION));

    options.addOption(OptionBuilder.withLongOpt("bin-folds-count").withDescription("binarization precision: how many binary features inferred from real one").hasArg().create(BIN_FOLDS_COUNT_OPTION));
    options.addOption(OptionBuilder.withLongOpt("grid").withDescription("file with already precomputed grid").hasArg().create(GRID_OPTION));
    options.addOption(OptionBuilder.withLongOpt("optimization").withDescription("optimization scheme: Strong/Weak or just Strong (" + DEFAULT_OPTIMIZATION_SCHEME + ")").hasArg().create(OPTIMIZATION_OPTION));

    options.addOption(OptionBuilder.withLongOpt("out").withDescription("output file name").hasArg().create(OUTPUT_OPTION));
    options.addOption(OptionBuilder.withLongOpt("matrixnetbin").withDescription("write model in matrix-net bin format").hasArg(false).create(WRITE_BIN_FORMULA));

    options.addOption(OptionBuilder.withLongOpt("verbose").withDescription("verbose output").create(VERBOSE_OPTION));
    options.addOption(OptionBuilder.withLongOpt("print-period").withDescription("number of iterations to evaluate and print scores").hasArg().create(PRINT_PERIOD));
    options.addOption(OptionBuilder.withLongOpt("fast-run").withDescription("fast run without model evaluation").create(FAST_OPTION));
    options.addOption(OptionBuilder.withLongOpt("histogram").withDescription("histogram for dynamic grid").hasArg(false).create(HIST_OPTION));

    options.addOption(OptionBuilder.withLongOpt("model").withDescription("model file").hasArg().create(MODEL_OPTION));
  }

  public static void main(final String[] args) throws IOException {
    final CommandLineParser parser = new GnuParser();
    try {
      final CommandLine command = parser.parse(options, args);
      if (command.getArgs().length == 0)
        throw new RuntimeException("Please provide mode to run");

      final String mode = command.getArgs()[0];
      switch (mode) {
        case "fit":
          modeFit(command);
          break;
        case "apply":
          modeApply(command);
          break;
        case "convert-pool":
          modeConvertPool(command);
          break;
        case "convert-pool-json2classic":
          modeConvertPoolJson2Classic(command);
          break;
        case "convert-pool-libfm":
          modeConvertPoolLibfm(command);
          break;
        case "validate-model":
          modeValidateModel(command);
          break;
        case "validate-pool":
          modeValidatePool(command);
          break;
        case "split-json-pool":
          modeSplitJsonPool(command);
          break;
        case "print-pool-info":
          modePrintPoolInfo(command);
          break;
        default:
          throw new RuntimeException("Mode " + mode + " is not recognized");
      }
    } catch (ParseException e) {
      System.err.println(e.getLocalizedMessage());
      final HelpFormatter formatter = new HelpFormatter();
      final String columns = System.getenv("COLUMNS");
      formatter.printUsage(new PrintWriter(System.err), columns != null ? Integer.parseInt(columns) : 80, "jmll", options);
      formatter.printHelp("JMLLCLI", options);

    } catch (Exception e) {
      e.printStackTrace();
      System.err.println(e.getLocalizedMessage());
    }
  }

  private static void modeFit(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }

    //data loading
    final DataBuilder dataBuilder;
    if (command.hasOption(CROSS_VALIDATION_OPTION)) {
      final DataBuilderCrossValidation dataBuilderCrossValidation = new DataBuilderCrossValidation();
      final String[] cvOptions = StringUtils.split(command.getOptionValue(CROSS_VALIDATION_OPTION), "/", 2);
      dataBuilderCrossValidation.setRandomSeed(Integer.valueOf(cvOptions[0]));
      dataBuilderCrossValidation.setPartition(Double.valueOf(cvOptions[1]));
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
      if (loss instanceof BlockwiseMLLLogit) {
        final int printPeriod = Integer.valueOf(command.getOptionValue(PRINT_PERIOD, "20"));
        progressPrinter = new MulticlassProgressPrinter(learn, test, printPeriod); //f*ck you with your custom different-dimensional metrics
      } else {
        progressPrinter = new DefaultProgressPrinter(learn, test, loss, metrics);
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
    if (!command.hasOption(FAST_OPTION)) {
      ResultsPrinter.printResults(result, learn, test, loss, metrics);
      if (loss instanceof BlockwiseMLLLogit) {
        ResultsPrinter.printMulticlassResults(result, learn, test);
      } else if (loss instanceof ClassicMulticlassLoss) {
        final int printPeriod = Integer.valueOf(command.getOptionValue(PRINT_PERIOD, "20"));
        MCTools.makeOneVsRestReport(learn, test, (JoinedBinClassModel) result, printPeriod);
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

  private static void modeApply(final CommandLine command) throws MissingArgumentException, IOException, ClassNotFoundException {
    if (!command.hasOption(LEARN_OPTION) || !command.hasOption(MODEL_OPTION)) {
      throw new MissingArgumentException("Please, provide 'LEARN_OPTION' and 'MODEL_OPTION'");
    }

    final DataBuilderClassic dataBuilder = new DataBuilderClassic();
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));
    final Pool pool = dataBuilder.create().getFirst();
    final VecDataSet vecDataSet = pool.vecData();

    final ModelsSerializationRepository serializationRepository;
    if (command.hasOption(GRID_OPTION)) {
      final GridBuilder gridBuilder = new GridBuilder();
      gridBuilder.setGrid(BFGrid.CONVERTER.convertFrom(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION)))));
      serializationRepository = new ModelsSerializationRepository(gridBuilder.create());
    } else {
      serializationRepository = new ModelsSerializationRepository();
    }

    try (final OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(getOutputName(command) + ".values"))) {
      final Computable model = DataTools.readModel(command.getOptionValue(MODEL_OPTION, "features.txt.model"), serializationRepository);
      final CharSeqBuilder value = new CharSeqBuilder();

      for (int i = 0; i < pool.size(); i++) {
        value.clear();
//        value.append(MathTools.CONVERSION.convert(vecDataSet.parent().at(i), CharSequence.class));
//        value.append('\t');
        value.append(MathTools.CONVERSION.convert(vecDataSet.at(i), CharSequence.class));
        value.append('\t');
        value.append(MathTools.CONVERSION.convert(model.compute(vecDataSet.at(i)), CharSequence.class));
        writer.append(value).append('\n');
      }
    }
  }

  private static void modeValidateModel(final CommandLine command) throws MissingArgumentException, IOException {
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

  private static void modeConvertPool(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }

    final DataBuilder dataBuilder = new DataBuilderClassic();
    dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    final Pool pool = dataBuilder.create().getFirst();
    final String outputName = command.hasOption(OUTPUT_OPTION) ? getOutputName(command) : getOutputName(command) + ".pool";
    DataTools.writePoolTo(pool, new FileWriter(outputName));
  }

  private static void modeConvertPoolJson2Classic(final CommandLine command) throws MissingArgumentException, IOException {
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

  private static void modeConvertPoolLibfm(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION");
    }

    final DataBuilderClassic dataBuilder = new DataBuilderClassic();
    dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    final Pool<?> pool = dataBuilder.create().getFirst();
    final String outputName = command.hasOption(OUTPUT_OPTION) ? getOutputName(command) : getOutputName(command) + ".libfm";
    try (final BufferedWriter out = new BufferedWriter(new FileWriter(outputName))) {
      DataTools.writePoolInLibfmFormat(pool, out);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static void modeValidatePool(final CommandLine command) throws MissingArgumentException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide 'LEARN_OPTION'");
    }

    final DataBuilder dataBuilder = new DataBuilderClassic();
    dataBuilder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    dataBuilder.setJsonFormat(command.hasOption(JSON_FORMAT));
    try {
      final Pool pool = dataBuilder.create().getFirst();
      System.out.println("Valid pool");
    } catch (Exception e) {
      System.out.println("Invalid pool: can't even load");
    }
  }

  private static void modeSplitJsonPool(final CommandLine command) throws MissingArgumentException, IOException {
    if (!command.hasOption(LEARN_OPTION) && !command.hasOption(JSON_FORMAT) && !command.hasOption(CROSS_VALIDATION_OPTION)) {
      throw new MissingArgumentException("Please provide: learn_option, json_flag and cross_validation_option");
    }

    final DataBuilderCrossValidation builder = new DataBuilderCrossValidation();
    builder.setJsonFormat(command.hasOption(JSON_FORMAT));
    builder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    final String[] cvOptions = StringUtils.split(command.getOptionValue(CROSS_VALIDATION_OPTION), "/", 2);
    builder.setRandomSeed(Integer.valueOf(cvOptions[0]));
    builder.setPartition(Double.valueOf(cvOptions[1]));

    final Pair<? extends Pool, ? extends Pool> pools = builder.create();

    final String outputName = getOutputName(command);
    DataTools.writePoolTo(pools.getFirst(), new FileWriter(outputName + ".learn"));
    DataTools.writePoolTo(pools.getSecond(), new FileWriter(outputName + ".test"));
  }

  private static void modePrintPoolInfo(final CommandLine command) throws MissingArgumentException {
    if (!command.hasOption(LEARN_OPTION)) {
      throw new MissingArgumentException("Please provide: learn_option");
    }

    final DataBuilder builder = new DataBuilderClassic();
    builder.setLearnPath(command.getOptionValue(LEARN_OPTION));
    builder.setJsonFormat(command.hasOption(JSON_FORMAT));
    final Pool<?> pool = builder.create().getFirst();
    System.out.println(DataTools.getPoolInfo(pool));
  }

  private static String getOutputName(final CommandLine command) {
    final String outName;
    if (command.hasOption(OUTPUT_OPTION)) {
      outName = command.getOptionValue(OUTPUT_OPTION);
    } else {
      final String tempName = command.getOptionValue(LEARN_OPTION, "features.txt");
      if (tempName.endsWith(".gz"))
        outName = tempName.substring(0, tempName.length() - ".gz".length());
      else
        outName = tempName;
    }
    return outName;
  }
}
