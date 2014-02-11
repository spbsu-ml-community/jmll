package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.func.WeakListenerHolder;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.text.StringUtils;
import com.spbsu.commons.util.Pair;
import com.spbsu.commons.util.logging.Interval;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;

import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.apache.commons.cli.*;

import java.io.*;
import java.lang.reflect.Method;
import java.util.Random;
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
  }

  public static void main(String[] args) throws IOException {
    CommandLineParser parser = new GnuParser();
    Random rnd = new FastRandom();
    ModelsSerializationRepository serializationRepository = new ModelsSerializationRepository();

    try {
      final CommandLine command = parser.parse(options, args);
      if (command.hasOption(GRID_OPTION))
        serializationRepository = serializationRepository.customizeGrid(serializationRepository.read(StreamTools.readFile(new File(command.getOptionValue('g'))), BFGrid.class));

      String learnFile = command.getOptionValue(LEARN_OPTION, "features.txt");

      final TIntObjectHashMap<CharSequence> metaLearn = new TIntObjectHashMap<CharSequence>();
      final TIntObjectHashMap<CharSequence> metaTest = new TIntObjectHashMap<CharSequence>();
      DataSet learn = DataTools.loadFromFeaturesTxt(learnFile, metaLearn);
      if (command.hasOption(NORMALIZE_RELEVANCE_OPTION))
        learn = DataTools.normalizeClasses(learn);
      if (learnFile.endsWith(".gz"))
        learnFile = learnFile.substring(0, learnFile.length() - ".gz".length());

      DataSet test;
      if (!command.hasOption(CROSS_VALIDATION_OPTION)) {
        test = command.hasOption(TEST_OPTION) ? DataTools.loadFromFeaturesTxt(command.getOptionValue(TEST_OPTION), metaTest) : learn;
      } else {
        final String cvOption = command.getOptionValue(CROSS_VALIDATION_OPTION);
        final String[] cvOptionsSplit = StringUtils.split(cvOption, "/", 2);
        final Pair<DataSet, DataSet> learnTest = splitCV(learn, Integer.parseInt(cvOptionsSplit[1]), new FastRandom(Integer.parseInt(cvOptionsSplit[0])));
        learn = learnTest.first;
        test = learnTest.second;
      }
      if (command.getArgs().length <= 0)
        throw new RuntimeException("Please provide mode to run");
      if (command.hasOption(NORMALIZE_RELEVANCE_OPTION))
        test = DataTools.normalizeClasses(test);

      String mode = command.getArgs()[0];

      final DataSet finalLearn = learn;
      final Factory<BFGrid> lazyGrid = new Factory<BFGrid>() {
        BFGrid cooked;

        @Override
        public BFGrid create() {
          if (cooked == null) {
            if (!command.hasOption("g"))
              cooked = GridTools.medianGrid(finalLearn, Integer.parseInt(command.getOptionValue(BIN_FOLDS_COUNT_OPTION, "32")));
            else
              cooked = BFGrid.CONVERTER.convertFrom(command.getOptionValue("g"));
          }
          return cooked;
        }
      };

      final String outputFile = command.hasOption(OUTPUT_OPTION) ? command.getOptionValue(OUTPUT_OPTION) : learnFile;

      if ("fit".equals(mode)) {
        final Optimization method = chooseMethod(command.getOptionValue(OPTIMIZATION_OPTION, DEFAULT_OPTIMIZATION_SCHEME), lazyGrid, rnd);
        final String target = command.getOptionValue(TARGET_OPTION, DEFAULT_TARGET);
        final Func loss = DataTools.targetByName(target).compute(learn.target());
        final String[] metricNames = command.getOptionValues(METRICS_OPTION);
        final Trans[] metrics = new Trans[metricNames != null ? metricNames.length : 1];
        if (metricNames != null) {
          for (int i = 0; i < metricNames.length; i++) {
            metrics[i] = DataTools.targetByName(metricNames[i]).compute(test.target());
          }
        } else {
          metrics[0] = loss;
        }

        final Action<Trans> progressHandler = new ProgressPrinter(learn, test, loss, metrics);
        if (method instanceof WeakListenerHolder && command.hasOption(VERBOSE_OPTION)) {
          //noinspection unchecked
          ((WeakListenerHolder) method).addListener(progressHandler);
        }

        Interval.start();
        Interval.suspend();
        @SuppressWarnings("unchecked")
        final Trans result = method.fit(learn, loss);
        Interval.stopAndPrint("Total fit time:");

        System.out.print("Learn: " + loss.value(result.transAll(learn.data())) + " Test:");
        for (final Trans metric : metrics) {
          System.out.print(" " + metric.trans(result.transAll(test.data())));
        }
        System.out.println();

        BFGrid grid = DataTools.grid(result);
        serializationRepository = serializationRepository.customizeGrid(grid);
        DataTools.writeModel(result, new File(outputFile + ".model"), serializationRepository);
        StreamTools.writeChars(serializationRepository.write(grid), new File(outputFile + ".grid"));
      } else if ("apply".equals(mode)) {
        OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(outputFile + ".values"));
        try {
          Trans model = DataTools.readModel(command.getOptionValue(MODEL_OPTION, "features.txt.model"), serializationRepository);
          DSIterator it = learn.iterator();
          int index = 0;
          while (it.advance()) {
            final CharSequence value;
            if (model instanceof Func) {
              final Func func = (Func) model;
              value = Double.toString(func.value(it.x()));
            } else {
              value = model.trans(it.x()).toString();
            }
            writer.append(metaLearn.get(index));
            writer.append('\t');
            writer.append(value);
            writer.append('\n');
            index++;
          }
        } finally {
          writer.close();
        }
      } else {
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


  private static Pair<DataSet, DataSet> splitCV(DataSet learn, int folds, Random rnd) {
    TIntArrayList learnIndices = new TIntArrayList();
    TIntArrayList testIndices = new TIntArrayList();
    for (int i = 0; i < learn.power(); i++) {
      if (rnd.nextDouble() < 1. / folds)
        learnIndices.add(i);
      else
        testIndices.add(i);
    }
    final int[] learnIndicesArr = learnIndices.toArray();
    final int[] testIndicesArr = testIndices.toArray();
    return Pair.<DataSet, DataSet>create(
        new DataSetImpl(
            new VecBasedMx(
                learn.xdim(),
                new IndexTransVec(
                    learn.data(),
                    new RowsPermutation(learnIndicesArr, learn.xdim())
                )
            ),
            new IndexTransVec(learn.target(), new ArrayPermutation(learnIndicesArr))
        ),
        new DataSetImpl(
            new VecBasedMx(
                learn.xdim(),
                new IndexTransVec(
                    learn.data(),
                    new RowsPermutation(testIndicesArr, learn.xdim())
                )
            ),
            new IndexTransVec(learn.target(), new ArrayPermutation(testIndicesArr))));
  }

  private static Optimization chooseMethod(String scheme, Factory<BFGrid> grid, Random rnd) {
    final int parametersStart = scheme.indexOf('(') >= 0 ? scheme.indexOf('(') : scheme.length();
    final Factory<? extends Optimization> factory = methodBuilderByName(scheme.substring(0, parametersStart), grid, rnd);
    String parameters = parametersStart < scheme.length() ? scheme.substring(parametersStart + 1, scheme.lastIndexOf(')')) : "";
    StringTokenizer paramsTok = new StringTokenizer(parameters, ",");
    Method[] builderMethods = factory.getClass().getMethods();
    while (paramsTok.hasMoreTokens()) {
      final String[] temp = StringUtils.split(paramsTok.nextToken().trim(), "=", 2);
      final String name = temp[0].trim();
      final String value = temp[1].trim();
      Method setter = null;
      for (int m = 0; m < builderMethods.length && setter == null; m++) {
        if (builderMethods[m].getName().equalsIgnoreCase(name))
          setter = builderMethods[m];
      }
      if (setter == null || setter.getParameterTypes().length > 1 || setter.getParameterTypes().length < 1) {
        System.err.println("Can not set up parameter: " + name + " to value: " + value + ". No setter in builder.");
        continue;
      }
      Class type = setter.getParameterTypes()[0];
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
          System.err.println("Can not set up parameter: " + name + " to value: " + value + ". Unknown parameter type: " + type + ".");
        }
      } catch (Exception e) {
        throw new RuntimeException("Can not set up parameter: " + name + " to value: " + value + ".", e);
      }
    }
    return factory.create();
  }

  private static Factory<? extends Optimization> methodBuilderByName(String name, final Factory<BFGrid> grid, final Random rnd) {
    if ("GradientBoosting".equals(name)) {
      return new Factory<Optimization>() {
        public Optimization weak = new BootstrapOptimization(new GreedyObliviousTree(grid.create(), 6), rnd);
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

        public void weak(Optimization weak) {
          this.weak = weak;
        }

        @Override
        public Optimization create() {
          //noinspection unchecked
          return new GradientBoosting(weak, DataTools.targetByName(lossName), icount, step);
        }
      };
    } else if ("MultiClassSplit".equals(name)) {
      return new Factory<MultiClass>() {
        public Optimization inner = new BootstrapOptimization(new GreedyObliviousTree(grid.create(), 6), rnd);
        public String localName = "SatL2";

        public void inner(Optimization i) {
          this.inner = i;
        }

        public void local(String localName) {
          this.localName = localName;
        }

        @Override
        public MultiClass create() {
          return new MultiClass(inner, (Computable<Vec, L2>) DataTools.targetByName(localName));
        }
      };
    } else if ("GreedyObliviousTree".equals(name)) {
      return new Factory<Optimization>() {
        public int depth = 6;

        public void depth(int d) {
          this.depth = d;
        }

        @Override
        public Optimization create() {
          return new GreedyObliviousTree(grid.create(), depth);
        }
      };
    } else if ("GreedyTDRegion".equals(name)) {
      return new Factory<Optimization>() {
        @Override
        public Optimization create() {
          return new GreedyTDRegion(grid.create());
        }
      };
    } else throw new RuntimeException("Unknown weak model: " + name);
  }


  private static class ProgressPrinter implements ProgressHandler {
    private final DataSet learn;
    private final DataSet test;
    private final Trans loss;
    private final Trans[] testMetrics;
    Vec learnValues;
    Vec[] testValuesArray;

    public ProgressPrinter(DataSet learn, DataSet test, Trans learnMetric, Trans[] testMetrics) {
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
        append(learnValues, VecTools.scale(last.transAll(learn.data()), step));
        for (final Vec testValues : testValuesArray) {
          append(testValues, VecTools.scale(last.transAll(test.data()), step));
        }
      } else {
        learnValues = partial.transAll(learn.data());
        for (int i = 0; i < testValuesArray.length; i++) {
          testValuesArray[i] = partial.transAll(test.data());
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
