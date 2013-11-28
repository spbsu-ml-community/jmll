package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.func.WeakListenerHolder;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.IntBasis;
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
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.func.Ensemble;
import gnu.trove.TIntArrayList;
import gnu.trove.TIntObjectHashMap;
import org.apache.commons.cli.*;

import java.io.*;
import java.lang.reflect.Method;
import java.util.Random;
import java.util.StringTokenizer;

import static com.spbsu.commons.math.vectors.VecTools.append;

/**
 * User: solar
 * Date: 08.08.13
 * Time: 17:38
 */
@SuppressWarnings("UnusedDeclaration")
public class JMLLCLI {
  public static final String DEFAULT_TARGET = "L2";
  public static final String DEFAULT_OPTIMIZATION_SCHEME = "GradientBoosting(local=SatL2, weak=GreedyObliviousTree(depth=6), step=0.02, iterations=1000)";
  static Options options = new Options();

  static {
    options.addOption("f", "learn", true, "features.txt format file used as learn");
    options.addOption("t", "test", true, "test part in features.txt format");
    options.addOption("T", "target", true, "target function to optimize format Global/Weak/Cursor (" + DEFAULT_TARGET + ")");
    options.addOption("M", "measure", true, "metric to test, by default equals to global optimization target");
    options.addOption("O", "optimization", true, "optimization scheme: Strong/Weak or just Strong (" + DEFAULT_OPTIMIZATION_SCHEME + ")");
    options.addOption("x", "bin-folds-count", true, "binarization precision: how many binary features inferred from real one");
    options.addOption("g", "grid", true, "file with already precomputed grid");
    options.addOption("m", "model", true, "model file");
    options.addOption("v", "verbose", false, "verbose output");
    options.addOption("r", "normalize-relevance", false, "relevance to classes");
    options.addOption("a", "forest-length", true, "forest length");
    options.addOption("X", "cross-validation", true, "k folds CV");
  }

  public static void main(String[] args) throws IOException {
    CommandLineParser parser = new GnuParser();
    Random rnd = new FastRandom();
    ModelsSerializationRepository serializationRepository = new ModelsSerializationRepository();

    try {
      final CommandLine command = parser.parse(options, args);
      if (command.hasOption('g'))
        serializationRepository = serializationRepository.customizeGrid(serializationRepository.read(StreamTools.readFile(new File(command.getOptionValue('g'))), BFGrid.class));

      String learnFile = command.getOptionValue("f", "features.txt");

      final TIntObjectHashMap<CharSequence> metaLearn = new TIntObjectHashMap<CharSequence>();
      final TIntObjectHashMap<CharSequence> metaTest = new TIntObjectHashMap<CharSequence>();
      DataSet learn = DataTools.loadFromFeaturesTxt(learnFile, metaLearn);
      if (command.hasOption('r'))
        learn = DataTools.normalizeClasses(learn);
      if (learnFile.endsWith(".gz"))
        learnFile = learnFile.substring(0, learnFile.length() - ".gz".length());

      DataSet test;
      if (!command.hasOption("X")){
        test = command.hasOption('t') ? DataTools.loadFromFeaturesTxt(command.getOptionValue('t'), metaTest) : learn;
      }
      else {
        final String cvOption = command.getOptionValue('X');
        final String[] cvOptionsSplit = StringUtils.split(cvOption, "/", 2);
        final Pair<DataSet, DataSet> learnTest = splitCV(learn, Integer.parseInt(cvOptionsSplit[1]), new FastRandom(Integer.parseInt(cvOptionsSplit[0])));
        learn = learnTest.first;
        test = learnTest.second;
      }
      if (command.getArgs().length <= 0)
        throw new RuntimeException("Please provide mode to run");
      if (command.hasOption('r'))
        test = DataTools.normalizeClasses(test);

      String mode = command.getArgs()[0];

      final DataSet finalLearn = learn;
      final Factory<BFGrid> lazyGrid = new Factory<BFGrid>() {
        BFGrid cooked;
        @Override
        public BFGrid create() {
          if (cooked == null) {
            if (!command.hasOption("g"))
              cooked = GridTools.medianGrid(finalLearn, Integer.parseInt(command.getOptionValue("x", "32")));
            else
              cooked = BFGrid.CONVERTER.convertFrom(command.getOptionValue("g"));
          }
          return cooked;
        }
      };

      if ("fit".equals(mode)) {
        final Optimization method = chooseMethod(command.getOptionValue("O", DEFAULT_OPTIMIZATION_SCHEME), lazyGrid, rnd);
        final String target = command.getOptionValue("T", DEFAULT_TARGET);
        final Func loss = DataTools.targetByName(target).compute(learn.target());
        final Func metric = DataTools.targetByName(command.getOptionValue("M", target)).compute(test.target());

        final Action<Trans> progressHandler = new ProgressPrinter(learn, test, loss, metric);
        if (method instanceof WeakListenerHolder && command.hasOption("v")) {
          //noinspection unchecked
          ((WeakListenerHolder)method).addListener(progressHandler);
        }
        Interval.start();
        Interval.suspend();
        @SuppressWarnings("unchecked")
        final Trans result = method.fit(learn, loss);
        Interval.stopAndPrint("Total fit time:");
        System.out.println("Learn: " + loss.value(result.transAll(learn.data())) + " Test: " + metric.value(result.transAll(test.data())));

        BFGrid grid = DataTools.grid(result);
        serializationRepository = serializationRepository.customizeGrid(grid);
        DataTools.writeModel(result, new File(learnFile + ".model"), serializationRepository);
        StreamTools.writeChars(serializationRepository.write(grid), new File(learnFile + ".grid"));
      } else if ("apply".equals(mode)) {
        OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(learnFile + ".values"));
        try {
          Trans model = DataTools.readModel(command.getOptionValue('m', "features.txt.model"), serializationRepository);
          if (model instanceof Func) {
            final Func func = (Func) model;
            DSIterator it = learn.iterator();
            int index = 0;
            while (it.advance()) {
              writer.append(metaLearn.get(index));
              writer.append('\t');
              writer.append(Double.toString(func.value(it.x())));
              writer.append('\n');
              index++;
            }
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
      if (rnd.nextDouble() < 1./folds)
        learnIndices.add(i);
      else
        testIndices.add(i);
    }
    final int[] learnIndicesArr = learnIndices.toNativeArray();
    final int[] testIndicesArr = testIndices.toNativeArray();
    return Pair.<DataSet,DataSet>create(
            new DataSetImpl(
                    new VecBasedMx(
                            learn.xdim(),
                            new IndexTransVec(
                                    learn.data(),
                                    new RowsPermutation(learnIndicesArr, learn.xdim()),
                                    new IntBasis(learnIndicesArr.length * learn.xdim())
                            )
                    ),
                    new IndexTransVec(learn.target(), new ArrayPermutation(learnIndicesArr), new IntBasis(learnIndicesArr.length))
            ),
            new DataSetImpl(
                    new VecBasedMx(
                            learn.xdim(),
                            new IndexTransVec(
                                    learn.data(),
                                    new RowsPermutation(testIndicesArr, learn.xdim()),
                                    new IntBasis(testIndicesArr.length * learn.xdim())
                            )
                    ),
                    new IndexTransVec(learn.target(), new ArrayPermutation(testIndicesArr), new IntBasis(testIndicesArr.length))));
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
        }
        else if (Double.class.equals(type) || double.class.equals(type)) {
          setter.invoke(factory, Double.parseDouble(value));
        }
        else if (String.class.equals(type)) {
          setter.invoke(factory, value);
        }
        else if (Optimization.class.equals(type)) {
          setter.invoke(factory, chooseMethod(value, grid, rnd));
        }
        else {
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
          return new MultiClass(inner, (Computable<Vec, L2>)DataTools.targetByName(localName));
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
          return new GreedyTDRegion(rnd, grid.create());
        }
      };
    } else throw new RuntimeException("Unknown weak model: " + name);
  }


  private static class ProgressPrinter implements ProgressHandler {
    private final DataSet learn;
    private final DataSet test;
    private final Trans loss;
    private final Trans testMetric;
    Vec learnValues;
    Vec testValues;

    public ProgressPrinter(DataSet learn, DataSet test, Trans learnMetric, Trans testMetric) {
      this.learn = learn;
      this.test = test;
      this.loss = learnMetric;
      this.testMetric = testMetric;
      learnValues = new ArrayVec(learnMetric.xdim());
      testValues = new ArrayVec(testMetric.xdim());
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
        append(testValues, VecTools.scale(last.transAll(test.data()), step));
      } else {
        learnValues = partial.transAll(learn.data());
        testValues = partial.transAll(test.data());
      }
      iteration++;
      if (iteration % 10 != 0)
        return;
      System.out.print(iteration);
      System.out.println(" " + loss.trans(learnValues) + "\t" + testMetric.trans(testValues));
//      Interval.resume();
    }
  }
}
