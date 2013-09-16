package com.spbsu.ml;

import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.io.ModelsSerializationRepository;
import com.spbsu.ml.loss.*;
import com.spbsu.ml.methods.*;
import com.spbsu.ml.methods.trees.GreedyContinousObliviousRegressionTree;
import com.spbsu.ml.methods.trees.GreedyObliviousClassificationTree;
import com.spbsu.ml.methods.trees.GreedyObliviousMultiClassificationTree;
import com.spbsu.ml.methods.trees.GreedyObliviousRegressionTree;
import com.spbsu.ml.models.AdditiveModel;
import com.spbsu.ml.models.AdditiveMultiClassModel;
import gnu.trove.TIntObjectHashMap;
import org.apache.commons.cli.*;

import java.io.*;
import java.util.List;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.append;
import static com.spbsu.commons.math.vectors.VecTools.assign;
import static com.spbsu.commons.math.vectors.VecTools.scale;

/**
 * User: solar
 * Date: 08.08.13
 * Time: 17:38
 */
public class JMLLCLI {
  static Options options = new Options();

  static {
    options.addOption("f", "learn", true, "features.txt format file used as learn");
    options.addOption("t", "test", true, "test part in features.txt format");
    options.addOption("T", "target", true, "target funtion to optimize (MSE, LL, etc.)");
    options.addOption("M", "method", true, "optimization method (Boosting, etc.)");
    options.addOption("W", "weak-model", true, "weak model in case of ensemble method (CART, OT, etc.)");
    options.addOption("e", "weak-target", true, "weak model target");
    options.addOption("i", "iterations", true, "ensemble power (iterations count)");
    options.addOption("s", "step", true, "shrinkage parameter/weight in ensemble");
    options.addOption("x", "bin-folds-count", true, "binarization precision: how many binary features inferred from real one");
    options.addOption("d", "depth", true, "tree depth");
    options.addOption("g", "grid", true, "file with already precomputed grid");
    options.addOption("m", "model", true, "model file");
    options.addOption("v", "verbose", false, "verbose output");
    options.addOption("r", "normalize-relevance", false, "relevance to classes");
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

      DataSet test = command.hasOption('t') ? DataTools.loadFromFeaturesTxt(command.getOptionValue('t'), metaTest) : learn;
      if (command.getArgs().length <= 0)
        throw new RuntimeException("Please provide mode to run");
      if (command.hasOption('r'))
        test = DataTools.normalizeClasses(test);

      String mode = command.getArgs()[0];
      if ("fit".equals(mode)) {
        String target = command.getOptionValue("T", "MSE");
        final MLMethodOrder1 method = chooseMethod(command.getOptionValue("M", "Boosting"), command, rnd, learn);
        final Oracle1 loss = chooseTarget(learn, target);
        final Oracle1 metric = chooseTarget(test, target);
        final Model result;

        if (method instanceof MLMultiClassMethodOrder1) {
          final int classesCount = DataTools.countClasses(learn.target());
          final Oracle1[] classesLearn = new Oracle1[classesCount];
          final Oracle1[] classesTest = new Oracle1[classesCount];
          for (int c = 0; c < classesTest.length; c++) {
            Vec classLearnTarget = VecTools.fillIndices(VecTools.fill(new ArrayVec(learn.power()), -1.), DataTools.extractClass(learn.target(), c), 1);
            classesLearn[c] = chooseTarget(new ChangedTarget((DataSetImpl)learn, classLearnTarget), target);
            Vec classTestTarget = VecTools.fillIndices(VecTools.fill(new ArrayVec(test.power()), -1.), DataTools.extractClass(test.target(), c), 1);
            classesTest[c] = chooseTarget(new ChangedTarget((DataSetImpl)test, classTestTarget), target);
          }
          final ProgressHandler progressHandler = new MultiClassProgressPrinter(learn, test, classesLearn, classesTest);
          if (method instanceof ProgressOwner && command.hasOption("v")) {
            ((ProgressOwner) method).addProgressHandler(progressHandler);
          }
          result = method.fit(learn, loss);

          for (int i = 0; i < classesTest.length; i++) {
            System.out.println("Class " + i
                    + " Learn: " + classesLearn[i].value(((MultiClassModel)result).value(learn, i))
                    + " Test: " + classesTest[i].value(((MultiClassModel)result).value(test, i)));
          }
        }
        else {
          final ProgressHandler progressHandler = new ProgressPrinter(learn, test, loss, metric);
          if (method instanceof ProgressOwner && command.hasOption("v")) {
            ((ProgressOwner) method).addProgressHandler(progressHandler);
          }
          result = method.fit(learn, loss);

          System.out.println("Learn: " + loss.value(result.value(learn)) + " Test: " + metric.value(result.value(test)));
        }

        BFGrid grid = DataTools.grid(result);
        serializationRepository = serializationRepository.customizeGrid(grid);
        DataTools.writeModel(result, new File(learnFile + ".model"), serializationRepository);
        StreamTools.writeChars(serializationRepository.write(grid),
                               new File(learnFile + ".grid"));
      }
      else if ("apply".equals(mode)) {
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
        }
        finally {
          writer.close();
        }
      }
      else {
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

  private static MLMethodOrder1 chooseMethod(String name, CommandLine line, Random rnd, DataSet learn) {
    MLMethodOrder1 method;
    if ("GBoosting".equals(name)) {
      method = new GradientBoosting(chooseMethod(line.getOptionValue("W", "ORT"), line, rnd, learn),
                                    Integer.parseInt(line.getOptionValue("i", "1000")),
                                    Double.parseDouble(line.getOptionValue("s", "0.01")), rnd);
    }
    else if ("Boosting".equals(name)) {
      method = new Boosting(chooseMethod(line.getOptionValue("W", "ORT"), line, rnd, learn),
                            chooseTarget(learn, line.getOptionValue("e", "MSE")),
                            Integer.parseInt(line.getOptionValue("i", "1000")),
                            Double.parseDouble(line.getOptionValue("s", "0.01")), rnd);
    }
    else if ("MBoosting".equals(name)) {
      method = new MulticlassBoosting((MLMultiClassMethodOrder1)chooseMethod(line.getOptionValue("W", "OMCT"), line, rnd, learn),
                                      Integer.parseInt(line.getOptionValue("i", "1000")),
                                      Double.parseDouble(line.getOptionValue("s", "0.01")), rnd);
    }
    else if ("ORT".equals(name)) {
      BFGrid grid;
      if (!line.hasOption("g"))
        grid = GridTools.medianGrid(learn, Integer.parseInt(line.getOptionValue("x", "32")));
      else
        grid = BFGrid.CONVERTER.convertFrom(line.getOptionValue("g"));
      method = new GreedyObliviousRegressionTree(rnd, learn, grid, Integer.parseInt(line.getOptionValue("d", "6")));
    }
    else if ("OCRT".equals(name)) {
      BFGrid grid;
      if (!line.hasOption("g"))
        grid = GridTools.medianGrid(learn, Integer.parseInt(line.getOptionValue("x", "32")));
      else
        grid = BFGrid.CONVERTER.convertFrom(line.getOptionValue("g"));
      method = new GreedyContinousObliviousRegressionTree(rnd, learn, grid, Integer.parseInt(line.getOptionValue("d", "6")));
    }
    else if ("OMCT".equals(name)) {
      BFGrid grid;
      if (!line.hasOption("g"))
        grid = GridTools.medianGrid(learn, Integer.parseInt(line.getOptionValue("x", "32")));
      else
        grid = BFGrid.CONVERTER.convertFrom(line.getOptionValue("g"));
      method = new GreedyObliviousMultiClassificationTree(rnd, learn, grid, Integer.parseInt(line.getOptionValue("d", "6")));
    }
    else if ("OCT".equals(name)) {
      BFGrid grid;
      if (!line.hasOption("g"))
        grid = GridTools.medianGrid(learn, Integer.parseInt(line.getOptionValue("x", "32")));
      else
        grid = BFGrid.CONVERTER.convertFrom(line.getOptionValue("g"));
      method = new GreedyObliviousClassificationTree(rnd, learn, grid, Integer.parseInt(line.getOptionValue("d", "6")));
    }
    else throw new RuntimeException("Unknown target: " + name);
    return method;
  }

  private static Oracle1 chooseTarget(DataSet learn, String target) {
    Oracle1 loss;
    if ("MSE".equals(target)) {
      loss = new L2Loss(learn.target());
    }
    else if ("LL".equals(target)) {
      loss = new LogLikelihoodSigmoid(learn.target());
    }
    else if ("LLX2".equals(target)) {
      loss = new LogLikelihoodXSquare(learn.target());
    }
    else if ("F1".equals(target)) {
      loss = new FBetaSigmoid(learn.target(), 1.);
    }
    else if ("MLL".equals(target)) {
      loss = new MulticlassLogLikelyhood(learn.target());
    }
    else throw new RuntimeException("Unknown target: " + target);
    return loss;
  }

  private static class ProgressPrinter implements ProgressHandler {
    private final DataSet learn;
    private final DataSet test;
    private final Oracle1 loss;
    private final Oracle1 testMetric;
    Vec learnValues;
    Vec testValues;

    public ProgressPrinter(DataSet learn, DataSet test, Oracle1 learnMetric, Oracle1 testMetric) {
      this.learn = learn;
      this.test = test;
      this.loss = learnMetric;
      this.testMetric = testMetric;
      learnValues = new ArrayVec(learn.power());
      testValues = new ArrayVec(test.power());
    }

    int iteration = 0;

    @Override
    public void progress(Model partial) {
      if (partial instanceof AdditiveModel) {
        final List<Model> models = ((AdditiveModel) partial).models;
        final double step = ((AdditiveModel) partial).step;
        final Model last = models.get(models.size() - 1);
        append(learnValues, scale(last.value(learn), step));
        append(testValues, scale(last.value(test), step));
      }
      else {
        learnValues = partial.value(learn);
        testValues = partial.value(test);
      }
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
    public void progress(Model partial) {
      if (partial instanceof AdditiveMultiClassModel) {
        final List<MultiClassModel> models = ((AdditiveMultiClassModel) partial).models;
        final double step = ((AdditiveMultiClassModel) partial).step;
        final MultiClassModel last = models.get(models.size() - 1);
        for (int c = 0; c < learnValues.length; c++) {
          append(learnValues[c], scale(last.value(learn, c), step));
          append(testValues[c], scale(last.value(test, c), step));
        }
      }
      else {
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
