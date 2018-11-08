package com.expleague.fmcb;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.GridTools;
import com.expleague.ml.cli.output.printers.MulticlassProgressPrinter;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.*;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.multiclass.gradfac.FMCBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.MultiClassModel;
import org.apache.commons.cli.*;

import javax.xml.crypto.Data;
import java.io.*;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

public class FMCBoostingMain {
  private static Options options = new Options();

  static {
    options.addOption(Option.builder()
            .longOpt("train")
            .desc("Path to the train dataset")
            .hasArg()
            .argName("TRAIN")
            .type(String.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("test")
            .desc("Path to the test dataset")
            .hasArg()
            .argName("TEST")
            .type(String.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("model")
            .desc("Path to the model")
            .hasArg()
            .argName("MODEL")
            .type(String.class)
            .required()
            .build());
    options.addOption(Option.builder()
            .longOpt("gamma")
            .desc("Learning rate for StochasticALS")
            .hasArg()
            .argName("GAMMA")
            .type(Number.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("max_iter")
            .desc("Max iterations count for StochasticALS")
            .hasArg()
            .argName("MAX_ITER")
            .type(Integer.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("n_bins")
            .desc("Bin factor")
            .hasArg()
            .argName("N_BINS")
            .type(Integer.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("depth")
            .desc("Depth of the weak trees")
            .hasArg()
            .argName("DEPTH")
            .type(Integer.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("n_iter")
            .desc("Number of weak learners")
            .hasArg()
            .argName("N_ITER")
            .type(Integer.class)
            .required()
            .type(Number.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("step")
            .desc("Learning rate")
            .hasArg()
            .argName("STEP")
            .type(Number.class)
            .required()
            .build());
    options.addOption(Option.builder()
            .longOpt("train_pred")
            .desc("Path to the file with predictions on training dataset")
            .hasArg()
            .argName("TRAIN_PRED")
            .type(String.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("test_pred")
            .desc("Path to the file with predictions on test dataset")
            .hasArg()
            .argName("TEST_PRED")
            .type(String.class)
            .build());
  }

  private static Trans fit(final FMCBoosting boosting, final Pool<?> train, final Pool<?> test) {
    final VecDataSet vecDataSet = train.vecData();
    final BlockwiseMLLLogit globalLoss = train.target(BlockwiseMLLLogit.class);
    // final MulticlassProgressPrinter multiclassProgressPrinter = new MulticlassProgressPrinter(train, test);
    // boosting.addListener(multiclassProgressPrinter);
    long startTime = System.currentTimeMillis();
    final Ensemble ensemble = boosting.fit(vecDataSet, globalLoss);
    Interval.setStart(startTime);
    Interval.stopAndPrint(" training");

    final Trans joined = ensemble.last() instanceof FuncJoin ? MCTools.joinBoostingResult(ensemble) : ensemble;
    final MultiClassModel multiclassModel = new MultiClassModel(joined);

    final String learnResult = MCTools.evalModel(multiclassModel, train, "[LEARN] ", false);
    System.out.println(learnResult);

    return joined;
  }

  private static Vec eval(final Trans ensemble, final Pool<?> test, final String comment) {
    final MultiClassModel multiclassModel = new MultiClassModel(ensemble);

    Interval.start();
    final String testResult = MCTools.evalModel(multiclassModel, test, comment + " ", false);
    System.out.println(testResult);
    Interval.stopAndPrint(" evaluation on " + comment);
    return multiclassModel.bestClassAll(test.vecData().data());
  }

  private static void saveIntVec(final Vec data, final String path) throws Exception {
    final String result = data.stream().mapToObj(Math::round).map(Object::toString).collect(Collectors.joining(","));
    final PrintStream out = new PrintStream(new FileOutputStream(new File(path)));
    out.println(result);
    out.close();
  }

  public static void main(String[] args) throws Exception {
    CommandLineParser parser = new DefaultParser();
    try {
      CommandLine cmd = parser.parse(options, args);

      final String model = cmd.getOptionValue("model", null);
      final String trainPath = cmd.getOptionValue("train", null);
      final String testPath = cmd.getOptionValue("test", null);
      final int iterCount = Integer.parseInt(cmd.getOptionValue("n_iter"));
      final double step = Double.parseDouble(cmd.getOptionValue("step"));
      final double gamma = Double.parseDouble(cmd.getOptionValue("gamma", "100"));
      final int maxIter = Integer.parseInt(cmd.getOptionValue("max_iter", "2000"));
      final int depth = Integer.parseInt(cmd.getOptionValue("depth", "5"));
      final int binFactor = Integer.parseInt(cmd.getOptionValue("n_bins", "32"));
      final String trainPredPath = cmd.getOptionValue("train_pred", null);
      final String testPredPath = cmd.getOptionValue("test_pred", null);

      final FastRandom rng = new FastRandom(0);

      Pool<?> train = null;
      if (trainPath != null) {
        final FileInputStream file = new FileInputStream(trainPath);
        final InputStream in = testPath.endsWith("gz") ? new GZIPInputStream(file) : file;
        InputStreamReader trainReader = new InputStreamReader(in);
        train = DataTools.loadFromFeaturesTxt(trainPath, trainReader);
      }

      Pool<?> test = null;
      if (testPath != null) {
        final FileInputStream file = new FileInputStream(testPath);
        final InputStream in = testPath.endsWith("gz") ? new GZIPInputStream(file) : file;
        final InputStreamReader testReader = new InputStreamReader(in);
        test = DataTools.loadFromFeaturesTxt(testPath, testReader);
      }

      Trans ensemble = null;
      if (train == null && model != null) {
        ensemble = DataTools.readModel(new FileInputStream(new File(model)));
      }

      if (train == null && test == null) {
        throw new IllegalArgumentException("Either train or test dataset is required!");
      }

      if (train == null && ensemble == null) {
        throw new IllegalArgumentException("You should specify the model for test evaluation!");
      }

      if (train != null) {
        final FMCBoosting boosting = new FMCBoosting(
                new StochasticALS(rng, gamma, maxIter, new StochasticALS.Cache(600, 0.01, rng)),
                new GreedyObliviousTree<L2>(GridTools.medianGrid(train.vecData(), binFactor), depth),
                L2.class,
                iterCount,
                step
        );

        ensemble = fit(boosting, train, test);

        if (model != null) {
          DataTools.writeModel(ensemble, new FileOutputStream(new File(model)));
        }
      }

      if (test != null) {
        final Vec pred = eval(ensemble, test, "test");
        if (testPredPath != null) {
          saveIntVec(pred, testPredPath);
        }
      }

      if (train != null) {
        final Vec pred = eval(ensemble, train, "train");
        if (trainPredPath != null) {
          saveIntVec(pred, trainPredPath);
        }
      }
    } catch (Exception e) {
      throw e;
    }
  }
}
