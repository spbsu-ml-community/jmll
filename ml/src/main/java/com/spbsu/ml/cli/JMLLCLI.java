package com.spbsu.ml.cli;

import com.spbsu.ml.cli.modes.AbstractMode;
import com.spbsu.ml.cli.modes.impl.*;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.io.PrintWriter;

/**
 * User: qdeee
 * Date: 03.09.14
 */
@SuppressWarnings("UnusedDeclaration,AccessStaticViaInstance")
public class JMLLCLI {
  public static final String DEFAULT_TARGET = "L2";
  public static final String DEFAULT_OPTIMIZATION_SCHEME = "GradientBoosting(local=SatL2, weak=GreedyObliviousTree(depth=6), step=0.02, iterations=1000)";

  public static final String LEARN_OPTION = "f";
  public static final String JSON_FORMAT = "j";
  public static final String TEST_OPTION = "t";
  public static final String CROSS_VALIDATION_OPTION = "X";

  public static final String TARGET_OPTION = "T";
  public static final String METRICS_OPTION = "M";

  public static final String BIN_FOLDS_COUNT_OPTION = "x";
  public static final String GRID_OPTION = "g";
  public static final String OPTIMIZATION_OPTION = "O";

  public static final String VERBOSE_OPTION = "v";
  public static final String PRINT_PERIOD = "i";
  public static final String FAST_OPTION = "fast";
  public static final String SKIP_FINAL_EVAL_OPTION = "fastfinal";
  public static final String HIST_OPTION = "h";
  public static final String OUTPUT_OPTION = "o";
  public static final String WRITE_BIN_FORMULA = "mxbin";

  public static final String MODEL_OPTION = "m";

  public static final String RANGES_OPTION = "r";
  public static final String RANDOM_SEED_OPTION = "seed";

  private static Options options = new Options();

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
    options.addOption(OptionBuilder.withLongOpt("skip-final-eval").withDescription("skip model evaluation on last step (faster)").create(SKIP_FINAL_EVAL_OPTION));
    options.addOption(OptionBuilder.withLongOpt("histogram").withDescription("histogram for dynamic grid").hasArg(false).create(HIST_OPTION));

    options.addOption(OptionBuilder.withLongOpt("model").withDescription("model file").hasArg().create(MODEL_OPTION));

    options.addOption(OptionBuilder.withLongOpt("ranges").withDescription("parameters ranges").hasArg().create(RANGES_OPTION));
    options.addOption(OptionBuilder.withLongOpt("seed").withLongOpt("random seed").hasArg().create(RANGES_OPTION));
  }

  public static void main(final String[] args) throws IOException {
    final CommandLineParser parser = new GnuParser();
    try {
      final CommandLine command = parser.parse(options, args);
      if (command.getArgs().length == 0)
        throw new RuntimeException("Please provide mode to run");

      final String modeName = command.getArgs()[0];
      final AbstractMode mode = getMode(modeName);
      mode.run(command);

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

  private static AbstractMode getMode(final String mode) {
    switch (mode) {
      case "fit":                       return new Fit();
      case "apply":                     return new Apply();
      case "convert-pool":              return new ConvertPool();
      case "convert-pool-json2classic": return new ConvertPoolJson2Classsic();
      case "convert-pool-libfm":        return new ConvertPoolLibSvm();
      case "validate-model":            return new ValidateModel();
      case "validate-pool":             return new ValidatePool();
      case "split-json-pool":           return new SplitJsonPool();
      case "print-pool-info":           return new PrintPoolInfo();
      case "grid-search":               return new GridSearch();
      case "cross-validation":          return new CrossValidation();
      default:
        throw new RuntimeException("Mode " + mode + " is not recognized");
    }
  }
}
