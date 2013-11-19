package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.WeakListenerHolder;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.IntBasis;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.IndexTransVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.text.StringUtils;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.GradientL2Cursor;
import com.spbsu.ml.methods.BooBag;
import com.spbsu.ml.methods.Boosting;
import com.spbsu.ml.methods.GreedyTDRegion;
import com.spbsu.ml.methods.MLMethod;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.AdditiveModel;
import com.spbsu.ml.models.AdditiveMultiClassModel;
import gnu.trove.TIntArrayList;
import gnu.trove.TIntObjectHashMap;
import org.apache.commons.cli.*;

import java.io.*;
import java.lang.reflect.Constructor;
import java.util.List;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * User: solar
 * Date: 08.08.13
 * Time: 17:38
 */
public class JMLLCLI {
  public static final String DEFAULT_OPTIMIZATION_STRATEGY = "L2/SatL2/Gradient";
  public static final String DEFAULT_OPTIMIZATION_SCHEME = "Boosting/GOT";
  static Options options = new Options();

  static {
    options.addOption("f", "learn", true, "features.txt format file used as learn");
    options.addOption("t", "test", true, "test part in features.txt format");
    options.addOption("T", "target", true, "target function to optimize format Global/Weak/Cursor (" + DEFAULT_OPTIMIZATION_STRATEGY + ")");
    options.addOption("M", "measure", true, "metric to test, by default equals to global optimization target");
    options.addOption("O", "optimization", true, "optimization scheme: Strong/Weak or just Strong (" + DEFAULT_OPTIMIZATION_SCHEME + ")");
    options.addOption("i", "iterations", true, "ensemble power (iterations count)");
    options.addOption("s", "step", true, "shrinkage parameter/weight in ensemble");
    options.addOption("x", "bin-folds-count", true, "binarization precision: how many binary features inferred from real one");
    options.addOption("d", "depth", true, "tree depth");
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
      if ("fit".equals(mode)) {
        final MLMethod method = chooseMethod(command.getOptionValue("O", DEFAULT_OPTIMIZATION_SCHEME), command, rnd, learn);
        final String strategy = command.getOptionValue("T", DEFAULT_OPTIMIZATION_STRATEGY);
        final Oracle0 loss = chooseTarget(learn.target(), method, strategy);
        final Oracle0 metric = targetByName(command.getOptionValue("M", StringUtils.split(strategy, "/", 3)[0])).compute(test.target());

        final Action<Model> progressHandler = new ProgressPrinter(learn, test, loss, metric);
        if (method instanceof WeakListenerHolder && command.hasOption("v")) {
          ((WeakListenerHolder<Model>)method).addListener(progressHandler);
        }
        final Model result = method.fit(learn, loss);

        System.out.println("Learn: " + loss.value(result.value(learn)) + " Test: " + metric.value(result.value(test)));

        BFGrid grid = DataTools.grid(result);
        serializationRepository = serializationRepository.customizeGrid(grid);
        DataTools.writeModel(result, new File(learnFile + ".model"), serializationRepository);
        StreamTools.writeChars(serializationRepository.write(grid),
                new File(learnFile + ".grid"));
      } else if ("apply".equals(mode)) {
        OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(learnFile + ".values"));
        try {
          Model model = DataTools.readModel(command.getOptionValue('m', "features.txt.model"), serializationRepository);
          DSIterator it = learn.iterator();
          int index = 0;
          while (it.advance()) {
            writer.append(metaLearn.get(index));
            writer.append('\t');
            writer.append(Double.toString(model.value(it.x())));
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

  private static Oracle0 chooseTarget(Vec target, MLMethod method, String command) {
    String[] scheme = StringUtils.split(command, "/", 3);
    String[] defaultScheme = StringUtils.split(DEFAULT_OPTIMIZATION_SCHEME, "/", 3);
    if (scheme.length < defaultScheme.length) {
      String[] newScheme = new String[3];
      System.arraycopy(scheme, 0, newScheme, 0, scheme.length);
      System.arraycopy(defaultScheme, scheme.length, newScheme, scheme.length, defaultScheme.length - scheme.length);
      scheme = newScheme;
    }
    Oracle0 result = targetByName(scheme[0]).compute(target);
    if (method instanceof Boosting) { // need to boil down result target to cursor target
      Computable<Vec,Oracle0> weak = targetByName(scheme[1]);
      String cursorName = scheme[2];
      if ("Gradient".equals(cursorName)) {
        if (result instanceof Oracle1)
          result = new GradientL2Cursor((Oracle1)result, weak);
        else
          throw new RuntimeException("Unable to start gradient based method on Oracle0 global target type");
      }
    }
    return result;
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

  private static MLMethod chooseMethod(String name, final CommandLine command, Random rnd, final DataSet learn) {
    String[] scheme = StringUtils.split(name, "/", 3);
    String[] defaultScheme = StringUtils.split(DEFAULT_OPTIMIZATION_SCHEME, "/", 3);
    if (scheme.length < defaultScheme.length) {
      String[] newScheme = new String[2];
      System.arraycopy(scheme, 0, newScheme, 0, scheme.length);
      System.arraycopy(defaultScheme, scheme.length, newScheme, scheme.length, defaultScheme.length - scheme.length);
      scheme = newScheme;
    }
    name = scheme[0];
    MLMethod method;
    if ("Boosting".equals(name)) {
      method = new Boosting(chooseMethod(scheme[1], command, rnd, learn),
              Integer.parseInt(command.getOptionValue("i", "1000")),
              Double.parseDouble(command.getOptionValue("s", "0.01")), rnd);
    } else if ("BooBag".equals(name)) {
      method = new BooBag(chooseMethod(scheme[1], command, rnd, learn),
              Integer.parseInt(command.getOptionValue("i", "1000")),
              Integer.parseInt(command.getOptionValue("a", "5")),
              Double.parseDouble(command.getOptionValue("s", "0.01")), rnd);
    } else if ("GOT".equals(name)) {
      BFGrid grid;
      if (!command.hasOption("g"))
        grid = GridTools.medianGrid(learn, Integer.parseInt(command.getOptionValue("x", "32")));
      else
        grid = BFGrid.CONVERTER.convertFrom(command.getOptionValue("g"));
      method = new GreedyObliviousTree(grid, Integer.parseInt(command.getOptionValue("d", "6")));
    } else if ("GTDR".equals(name)) {
      BFGrid grid;
      if (!command.hasOption("g"))
        grid = GridTools.medianGrid(learn, Integer.parseInt(command.getOptionValue("x", "32")));
      else
        grid = BFGrid.CONVERTER.convertFrom(command.getOptionValue("g"));
      method = new GreedyTDRegion(rnd, learn, grid);
    } else throw new RuntimeException("Unknown weak model: " + name);
    return method;
  }


  private static Computable<Vec, Oracle0> targetByName(final String name) {
    try {
      Class<Oracle0> oracleClass = (Class<Oracle0>)Class.forName("com.spbsu.ml.loss." + name);
      final Constructor<Oracle0> constructor = oracleClass.getConstructor(Vec.class);
      return new Computable<Vec, Oracle0>() {
        @Override
        public Oracle0 compute(Vec argument) {
          try {
            return constructor.newInstance(argument);
          } catch (Exception e) {
            throw new RuntimeException("Exception during metric " + name + " initialization", e);
          }
        }
      };
    }
    catch (Exception e) {
      throw new RuntimeException("Unable to create requested target: " + name, e);
    }
  }

  private static class ProgressPrinter implements ProgressHandler {
    private final DataSet learn;
    private final DataSet test;
    private final Oracle0 loss;
    private final Oracle0 testMetric;
    Vec learnValues;
    Vec testValues;

    public ProgressPrinter(DataSet learn, DataSet test, Oracle0 learnMetric, Oracle0 testMetric) {
      this.learn = learn;
      this.test = test;
      this.loss = learnMetric;
      this.testMetric = testMetric;
      learnValues = new ArrayVec(learn.power());
      testValues = new ArrayVec(test.power());
    }

    int iteration = 0;

    @Override
    public void invoke(Model partial) {

      if (partial instanceof AdditiveModel) {
        final List<Model> models = ((AdditiveModel) partial).models;
        final double step = ((AdditiveModel) partial).step;
        final Model last = models.get(models.size() - 1);
        append(learnValues, scale(last.value(learn), step));
        append(testValues, scale(last.value(test), step));
      } else {
        learnValues = partial.value(learn);
        testValues = partial.value(test);
      }
      iteration++;
      if (iteration % 10 != 0)
        return;
      System.out.print(iteration++);
      System.out.println(" " + loss.value(learnValues) + "\t" + testMetric.value(testValues));
    }
  }

  private static class MultiClassProgressPrinter implements ProgressHandler {
    private final DataSet learn;
    private final DataSet test;
    private final Oracle1[] learnMetric;
    private final Oracle1[] testMetric;
    Vec[] learnValues;
    Vec[] testValues;

    public MultiClassProgressPrinter(DataSet learn, DataSet test, Oracle1[] learnMetric, Oracle1[] testMetric) {
      this.learn = learn;
      this.test = test;
      this.learnMetric = learnMetric;
      this.testMetric = testMetric;
      int classesCount = DataTools.countClasses(learn.target());

      learnValues = new Vec[classesCount];
      testValues = new Vec[classesCount];
      for (int i = 0; i < learnValues.length; i++) {
        learnValues[i] = new ArrayVec(learn.power());
        testValues[i] = new ArrayVec(test.power());
      }
    }

    private int iteration = 0;

    @Override
    public void invoke(Model partial) {
      if (partial instanceof AdditiveMultiClassModel) {
        final List<MultiClassModel> models = ((AdditiveMultiClassModel) partial).models;
        final double step = ((AdditiveMultiClassModel) partial).step;
        final MultiClassModel last = models.get(models.size() - 1);
        for (int c = 0; c < learnValues.length; c++) {
          append(learnValues[c], scale(last.value(learn, c), step));
          append(testValues[c], scale(last.value(test, c), step));
        }
      } else {
        for (int c = 0; c < learnValues.length; c++) {
          assign(learnValues[c], ((MultiClassModel) partial).value(learn, c));
          assign(testValues[c], ((MultiClassModel) partial).value(test, c));
        }
      }
      System.out.print(iteration++);
      for (int i = 0; i < learnValues.length; i++) {
        Vec learnValue = learnValues[i];
        Vec testValue = testValues[i];
        System.out.print("\t");
        System.out.print(i + ": " + learnMetric[i].value(learnValue) + " " + testMetric[i].value(testValue));
      }
      System.out.println();
    }
  }
}
