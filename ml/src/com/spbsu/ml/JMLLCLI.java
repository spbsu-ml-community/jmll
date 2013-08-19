package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.loss.LogLikelyhood;
import com.spbsu.ml.methods.Boosting;
import com.spbsu.ml.methods.GreedyObliviousTree;
import com.spbsu.ml.methods.MLMethodOrder1;
import com.spbsu.ml.methods.ProgressOwner;
import com.spbsu.ml.models.AdditiveModel;
import org.apache.commons.cli.*;

import java.io.PrintWriter;
import java.util.List;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.*;

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
    options.addOption("i", "iterations", true, "ensemble power (iterations count)");
    options.addOption("s", "step", true, "shrinkage parameter/weight in ensemble");
    options.addOption("x", "bin-folds-count", true, "binarization precision: how many binary features inferred from real one");
    options.addOption("d", "depth", true, "tree depth");
    options.addOption("g", "grid", true, "file with already precomputed grid");
    options.addOption("v", "verbose", false, "verbose output");
  }

  public static void main(String[] args) {
    CommandLineParser parser = new GnuParser();
    Random rnd = new FastRandom();
    try {
      final CommandLine command = parser.parse(options, args);
      final DataSet learn = DataTools.loadFromFeaturesTxt(command.getOptionValue("f", "features.txt"));
      final DataSet test = DataTools.loadFromFeaturesTxt(command.getOptionValue("t", "features.txt"));
      if (command.getArgs().length <= 0)
        throw new RuntimeException("Please provide mode to run");
      String mode = command.getArgs()[0];
      if ("fit".equals(mode)) {
        String target = command.getOptionValue("T", "MSE");
        final Oracle1 loss = chooseTarget(learn, target);
        final Oracle1 metric = chooseTarget(test, target);
        final MLMethodOrder1 method = chooseMethod(command.getOptionValue("M", "Boosting"), command, rnd, learn);
        final ProgressHandler progressHandler = new ProgressHandler() {
          Vec learnValues = new ArrayVec(learn.power());
          Vec testValues = new ArrayVec(test.power());
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
            System.out.println(loss.value(learnValues) + "\t" + metric.value(testValues));
          }
        };
        if (method instanceof ProgressOwner && command.hasOption("v")) {
          ((ProgressOwner) method).addProgressHandler(progressHandler);
        }
        final Model result = method.fit(learn, loss);

        System.out.println("Learn: " + loss.value(result.value(learn)) + " Test: " + metric.value(result.value(test)));
        System.out.println(result.toString());
      }
      else {
        throw new RuntimeException("Mode " + mode + " is not recognized");
      }
    } catch (Exception e) {
      HelpFormatter formatter = new HelpFormatter();
      System.err.println(e.getLocalizedMessage());
      String columns = System.getenv("COLUMNS");

      formatter.printUsage(new PrintWriter(System.err), columns != null ? Integer.parseInt(columns) : 80, "jmll", options);
    }
  }

  private static MLMethodOrder1 chooseMethod(String name, CommandLine line, Random rnd, DataSet learn) {
    MLMethodOrder1 method;
    if ("Boosting".equals(name)) {
      method = new Boosting(chooseMethod(line.getOptionValue("W", "OT"), line, rnd, learn),
                            Integer.parseInt(line.getOptionValue("i", "1000")),
                            Double.parseDouble(line.getOptionValue("s", "0.01")));
    }
    else if ("OT".equals(name)) {
      BFGrid grid;
      if (!line.hasOption("g"))
        grid = GridTools.medianGrid(learn, Integer.parseInt(line.getOptionValue("x", "32")));
      else
        grid = BFGrid.CONVERTER.convertFrom(line.getOptionValue("g"));
      method = new GreedyObliviousTree(rnd, learn, grid, Integer.parseInt(line.getOptionValue("d", "6")));
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
      loss = new LogLikelyhood(learn.target());
    }
    else throw new RuntimeException("Unknown target: " + target);
    return loss;
  }
}
