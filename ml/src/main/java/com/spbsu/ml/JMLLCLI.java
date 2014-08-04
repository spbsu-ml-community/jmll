package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.func.WeakListenerHolder;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqBuilder;
import com.spbsu.commons.text.StringUtils;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.logging.Interval;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.SubPool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import org.apache.commons.cli.*;
import org.jetbrains.annotations.Nullable;

import java.io.*;
import java.lang.reflect.Method;
import java.util.StringTokenizer;

import static com.spbsu.commons.math.vectors.VecTools.append;

/**
 * User: solar
 * User: starlight
 * Date: 08.08.13
 * Time: 17:38
 */
@SuppressWarnings("UnusedDeclaration,AccessStaticViaInstance")
public class JMLLCLI {
  public static final String DEFAULT_TARGET = "L2";
  public static final String DEFAULT_OPTIMIZATION_SCHEME = "GradientBoosting(local=SatL2, weak=GreedyObliviousTree(depth=6), step=0.02, iterations=1000)";

  private static final String LEARN_OPTION = "f";
  private static final String TEST_OPTION = "t";
  private static final String TARGET_OPTION = "T";
  private static final String METRICS_OPTION = "M";
  private static final String OPTIMIZATION_OPTION = "O";
  private static final String BIN_FOLDS_COUNT_OPTION = "x";
  private static final String GRID_OPTION = "g";
  private static final String MODEL_OPTION = "m";
  private static final String VERBOSE_OPTION = "v";
  private static final String NORMALIZE_RELEVANCE_OPTION = "r";
  private static final String FOREST_LENGTH_OPTION = "a";
  private static final String CROSS_VALIDATION_OPTION = "X";
  private static final String OUTPUT_OPTION = "o";
  private static final String JSON_FORMAT = "j";

  static Options options = new Options();

  static {
    options.addOption(OptionBuilder.withLongOpt("learn").withDescription("features.txt format file used as learn").hasArg().create(LEARN_OPTION));
    options.addOption(OptionBuilder.withLongOpt("test").withDescription("test part in features.txt format").hasArg().create(TEST_OPTION));
    options.addOption(OptionBuilder.withLongOpt("target").withDescription("target function to optimize format Global/Weak/Cursor (" + DEFAULT_TARGET + ")").hasArg().create(TARGET_OPTION));
    options.addOption(OptionBuilder.withLongOpt("metrics").withDescription("metrics to test, by default contains global optimization target").hasArgs().create(METRICS_OPTION));
    options.addOption(OptionBuilder.withLongOpt("optimization").withDescription("optimization scheme: Strong/Weak or just Strong (" + DEFAULT_OPTIMIZATION_SCHEME + ")").hasArg().create(OPTIMIZATION_OPTION));
    options.addOption(OptionBuilder.withLongOpt("bin-folds-count").withDescription("binarization precision: how many binary features inferred from real one").hasArg().create(BIN_FOLDS_COUNT_OPTION));
    options.addOption(OptionBuilder.withLongOpt("grid").withDescription("file with already precomputed grid").hasArg().create(GRID_OPTION));
    options.addOption(OptionBuilder.withLongOpt("model").withDescription("model file").hasArg().create(MODEL_OPTION));
    options.addOption(OptionBuilder.withLongOpt("verbose").withDescription("verbose output").create(VERBOSE_OPTION));
    options.addOption(OptionBuilder.withLongOpt("normalize-relevance").withDescription("relevance to classes").create(NORMALIZE_RELEVANCE_OPTION));
    options.addOption(OptionBuilder.withLongOpt("forest-length").withDescription("forest length").hasArg().create(FOREST_LENGTH_OPTION));
    options.addOption(OptionBuilder.withLongOpt("cross-validation").withDescription("k folds CV").hasArg().create(CROSS_VALIDATION_OPTION));
    options.addOption(OptionBuilder.withLongOpt("out").withDescription("output file name").hasArg().create(OUTPUT_OPTION));
    options.addOption(OptionBuilder.withLongOpt("json-format").withDescription("alternative format for features.txt").hasArg(false).create(JSON_FORMAT));
  }

  public static void main(String[] args) throws IOException {
    final CommandLineParser parser = new GnuParser();
    final ModelsSerializationRepository serializationRepository;

    try {
      final CommandLine command = parser.parse(options, args);

      if (command.getArgs().length == 0)
        throw new RuntimeException("Please provide mode to run");

      if (command.hasOption(GRID_OPTION)) {
        final BFGrid grid = DataTools.SERIALIZATION.read(StreamTools.readFile(new File(command.getOptionValue(GRID_OPTION))), BFGrid.class);
        serializationRepository = new ModelsSerializationRepository(grid);
      }
      else {
        serializationRepository = new ModelsSerializationRepository();
      }

      String learnFile = command.getOptionValue(LEARN_OPTION, "features.txt");

      Pool learn = command.hasOption(JSON_FORMAT) ? DataTools.loadFromFile(learnFile)
                                                  : DataTools.loadFromFeaturesTxt(learnFile);

//      if (command.hasOption(NORMALIZE_RELEVANCE_OPTION))
//        learn = MCTools.normalizeClasses(learn);

      if (learnFile.endsWith(".gz"))
        learnFile = learnFile.substring(0, learnFile.length() - ".gz".length());

      final Pool test;
      if (!command.hasOption(CROSS_VALIDATION_OPTION)) {
        test = command.hasOption(TEST_OPTION) ? DataTools.loadFromFeaturesTxt(command.getOptionValue(TEST_OPTION)) : learn;
      }
      else {
        final String cvOption = command.getOptionValue(CROSS_VALIDATION_OPTION);
        final String[] cvOptionsSplit = StringUtils.split(cvOption, "/", 2);
        final Pair<SubPool, SubPool> learnTest = splitCV(learn, Integer.parseInt(cvOptionsSplit[1]), new FastRandom(Integer.parseInt(cvOptionsSplit[0])));
        learn = learnTest.first;
        test = learnTest.second;
      }

      final String mode = command.getArgs()[0];
      final String outputFile = command.hasOption(OUTPUT_OPTION) ? command.getOptionValue(OUTPUT_OPTION) : learnFile;

      final VecDataSet learnDS = learn.vecData();
      switch (mode) {
        case "fit":
          final FastRandom rnd = new FastRandom();
          final Factory<BFGrid> lazyGrid = new Factory<BFGrid>() {
            BFGrid cooked;

            @Override
            public BFGrid create() {
              if (cooked == null) {
                if (command.hasOption(GRID_OPTION)) {
                  cooked = serializationRepository.getGrid();
                } else {
                  cooked = GridTools.medianGrid(learnDS, Integer.parseInt(command.getOptionValue(BIN_FOLDS_COUNT_OPTION, "32")));
                }
              }
              return cooked;
            }
          };
          final Optimization method = chooseMethod(command.getOptionValue(OPTIMIZATION_OPTION, DEFAULT_OPTIMIZATION_SCHEME), lazyGrid, rnd);

          final String target = command.getOptionValue(TARGET_OPTION, DEFAULT_TARGET);
          final TargetFunc loss = learn.target(DataTools.targetByName(target));

          final String[] metricNames = command.getOptionValues(METRICS_OPTION);
          final Func[] metrics;
          if (metricNames != null) {
            metrics = new Func[metricNames.length];
            for (int i = 0; i < metricNames.length; i++) {
              metrics[i] = test.target(DataTools.targetByName(metricNames[i]));
            }
          }
          else {
            metrics = new Func[] {
                test.target(DataTools.targetByName(target))
            };
          }

          final Action<Trans> progressHandler = new ProgressPrinter(learn, test, loss, metrics);
          if (method instanceof WeakListenerHolder && command.hasOption(VERBOSE_OPTION)) {
            //noinspection unchecked
            ((WeakListenerHolder) method).addListener(progressHandler);
          }


          Interval.start();
          Interval.suspend();

          @SuppressWarnings("unchecked")
          final Computable result = method.fit(learnDS, loss);

          Interval.stopAndPrint("Total fit time:");
          System.out.print("Learn: " + loss.value(DataTools.calcAll(result, learnDS)) + " Test:");
          for (final Trans metric : metrics) {
            System.out.print(" " + metric.trans(DataTools.calcAll(result, test.vecData())));
          }
          System.out.println();

          if (serializationRepository.getGrid() == null) {
            @Nullable BFGrid grid = DataTools.grid(result);
            if (grid != null) {
              StreamTools.writeChars(DataTools.SERIALIZATION.write(grid), new File(outputFile + ".grid"));
            }
          }
          DataTools.writeModel(result, new File(outputFile + ".model"));
          break;
        case "apply":
          try (final OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(outputFile + ".values"))) {
            final Computable model = DataTools.readModel(command.getOptionValue(MODEL_OPTION, "features.txt.model"), serializationRepository);
            final CharSeqBuilder value = new CharSeqBuilder();

            for (int i = 0; i < learnDS.length(); i++) {
              value.clear();
              value.append(MathTools.CONVERSION.convert(learnDS.parent().at(i), CharSequence.class));
              value.append('\t');
              value.append(MathTools.CONVERSION.convert(learnDS.at(i), CharSequence.class));
              value.append('\t');
              value.append(MathTools.CONVERSION.convert(model.compute(learnDS.at(i)), CharSequence.class));
              writer.append(value).append('\n');
            }
          }
          break;
        case "convert-pool":
          DataTools.writePoolTo(learn, new FileWriter(command.getOptionValue('o', "features.pool")));
          break;
        default:
          throw new RuntimeException("Mode " + mode + " is not recognized");
      }
    } catch (Exception e) {
      HelpFormatter formatter = new HelpFormatter();
      e.printStackTrace();
      System.err.println(e.getLocalizedMessage());
      String columns = System.getenv("COLUMNS");

      formatter.printUsage(new PrintWriter(System.err), columns != null ? Integer.parseInt(columns) : 80, "jmll", options);
    }
  }

  private static <I extends DSItem> Pair<SubPool<I>, SubPool<I>> splitCV(Pool<I> pool, int folds, FastRandom rnd) {
    final int[][] cvSplit = DataTools.splitAtRandom(pool.size(), rnd, 1. / folds, (folds - 1.) / folds);
    return Pair.create(new SubPool<I>(pool, cvSplit[0]), new SubPool<I>(pool, cvSplit[1]));
  }

  private static VecOptimization chooseMethod(String scheme, Factory<BFGrid> grid, FastRandom rnd) {
    final int parametersStart = scheme.indexOf('(') >= 0 ? scheme.indexOf('(') : scheme.length();
    final Factory<? extends VecOptimization> factory = methodBuilderByName(scheme.substring(0, parametersStart), grid, rnd);
    final String parameters = parametersStart < scheme.length() ? scheme.substring(parametersStart + 1, scheme.lastIndexOf(')')) : "";
    final StringTokenizer paramsTok = new StringTokenizer(parameters, ",");
    final Method[] builderMethods = factory.getClass().getMethods();
    while (paramsTok.hasMoreTokens()) {
      final String param = paramsTok.nextToken();
      final int splitPos = param.indexOf('=');
      final String name = param.substring(0, splitPos).trim();
      final String value = param.substring(splitPos + 1, param.length()).trim();
      Method setter = null;
      for (int m = 0; m < builderMethods.length && setter == null; m++) {
        if (builderMethods[m].getName().equalsIgnoreCase(name))
          setter = builderMethods[m];
      }
      if (setter == null || setter.getParameterTypes().length > 1 || setter.getParameterTypes().length < 1) {
        System.err.println("Can not set up parameter: " + name + " to value: " + value + ". No setter in builder.");
        continue;
      }
      final Class type = setter.getParameterTypes()[0];
      try {
        if (Integer.class.equals(type) || int.class.equals(type)) {
          setter.invoke(factory, Integer.parseInt(value));
        } else if (Double.class.equals(type) || double.class.equals(type)) {
          setter.invoke(factory, Double.parseDouble(value));
        } else if (String.class.equals(type)) {
          setter.invoke(factory, value);
        } else if (Optimization.class.equals(type)) {
          setter.invoke(factory, chooseMethod(value, grid, rnd));
        } else {
          System.err.println("Can not set up parameter: " + name + " to value: " + value + ". Unknown parameter type: " + type + "");
        }
      } catch (Exception e) {
        throw new RuntimeException("Can not set up parameter: " + name + " to value: " + value + "", e);
      }
    }
    return factory.create();
  }

  private static Factory<? extends VecOptimization> methodBuilderByName(String name, final Factory<BFGrid> grid, final FastRandom rnd) {
    switch (name) {
      case "GradientBoosting":
        return new Factory<VecOptimization>() {
          public VecOptimization weak = new BootstrapOptimization(new GreedyObliviousTree(grid.create(), 6), rnd);
          public String lossName = "LogL2";
          public double step = 0.005;
          public int icount = 200;

          public void step(double s) {
            this.step = s;
          }

          public void iterations(int icount) {
            this.icount = icount;
          }

          public void local(String lossName) {
            this.lossName = lossName;
          }

          public void weak(VecOptimization weak) {
            this.weak = weak;
          }

          @Override
          public VecOptimization create() {
            //noinspection unchecked
            return new GradientBoosting(weak, DataTools.targetByName(lossName), icount, step);
          }
        };
      case "MultiClassSplit":
        return new Factory<MultiClass>() {
          public VecOptimization inner = new BootstrapOptimization(new GreedyObliviousTree(grid.create(), 6), rnd);
          public String localName = "SatL2";

          public void inner(VecOptimization i) {
            this.inner = i;
          }

          public void local(String localName) {
            this.localName = localName;
          }

          @Override
          public MultiClass create() {
            return new MultiClass(inner, (Class<? extends L2>) DataTools.targetByName(localName));
          }
        };
      case "GreedyObliviousTree":
        return new Factory<VecOptimization>() {
          public int depth = 6;

          public void depth(int d) {
            this.depth = d;
          }

          @Override
          public VecOptimization create() {
            return new GreedyObliviousTree(grid.create(), depth);
          }
        };
      case "GreedyTDRegion":
        return new Factory<VecOptimization>() {
          @Override
          public VecOptimization create() {
            return new GreedyTDRegion(grid.create());
          }
        };
      case "FMWorkaround":
        return new Factory<VecOptimization>() {
          public String task = "-r";
          public String dim = "1,1,8";
          public String iters = "1000";
          public String others = "";

          public void task(final String task) {
            this.task = task;
          }

          public void dim(final String dim) {
            this.dim = dim;
          }

          public void iters(final String iters) {
            this.iters = iters;
          }

          public void others(final String others) {
            this.others = others;
          }

          @Override
          public VecOptimization create() {
            return new FMTrainingWorkaround(task, dim, iters, others);
          }
        };
      default:
        throw new RuntimeException("Unknown weak model: " + name);
    }
  }


  private static class ProgressPrinter implements ProgressHandler {
    private final Pool learn;
    private final Pool test;
    private final Func loss;
    private final Func[] testMetrics;
    Vec learnValues;
    Vec[] testValuesArray;

    public ProgressPrinter(Pool learn, Pool test, Func learnMetric, Func[] testMetrics) {
      this.learn = learn;
      this.test = test;
      this.loss = learnMetric;
      this.testMetrics = testMetrics;
      learnValues = new ArrayVec(learnMetric.xdim());
      testValuesArray = new ArrayVec[testMetrics.length];
      for (int i = 0; i < testValuesArray.length; i++) {
        testValuesArray[i] = new ArrayVec(testMetrics[i].xdim());
      }
    }

    int iteration = 0;

    @Override
    public void invoke(Trans partial) {
//      Interval.suspend();
      if (partial instanceof Ensemble) {
        final Ensemble ensemble = (Ensemble) partial;
        final double step = ensemble.wlast();
        final Trans last = ensemble.last();
        append(learnValues, VecTools.scale(last.transAll(learn.vecData().data()), step));
        for (final Vec testValues : testValuesArray) {
          append(testValues, VecTools.scale(last.transAll(test.vecData().data()), step));
        }
      } else {
        learnValues = partial.transAll(learn.vecData().data());
        for (int i = 0; i < testValuesArray.length; i++) {
          testValuesArray[i] = partial.transAll(test.vecData().data());
        }
      }
      iteration++;
      if (iteration % 10 != 0) {
        return;
      }

      System.out.print(iteration);
      System.out.print(" " + loss.trans(learnValues));
      for (int i = 0; i < testMetrics.length; i++) {
        System.out.print("\t" + testMetrics[i].trans(testValuesArray[i]));
      }
      System.out.println();
//      Interval.resume();
    }
  }
}
