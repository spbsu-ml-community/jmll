package com.expleague.fmcb;

import com.expleague.commons.math.Trans;
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

import java.io.*;
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
            .required()
            .build());
    options.addOption(Option.builder()
            .longOpt("test")
            .desc("Path to the test dataset")
            .hasArg()
            .argName("TEST")
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
            .longOpt("bin_factor")
            .desc("Bin factor")
            .hasArg()
            .argName("BIN_FACTOR")
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
            .longOpt("iter_count")
            .desc("Number of weak learners")
            .hasArg()
            .argName("ITER_COUNT")
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
  }

  private static Trans fit(final FMCBoosting boosting, final Pool<?> train, final Pool<?> test) {
    final VecDataSet vecDataSet = train.vecData();
    final BlockwiseMLLLogit globalLoss = train.target(BlockwiseMLLLogit.class);
    final MulticlassProgressPrinter multiclassProgressPrinter = new MulticlassProgressPrinter(train, test);
    boosting.addListener(multiclassProgressPrinter);

    Interval.start();
    final Ensemble ensemble = boosting.fit(vecDataSet, globalLoss);
    Interval.stopAndPrint(" trainging");

    final Trans joined = ensemble.last() instanceof FuncJoin ? MCTools.joinBoostingResult(ensemble) : ensemble;
    final MultiClassModel multiclassModel = new MultiClassModel(joined);

    final String learnResult = MCTools.evalModel(multiclassModel, train, "[LEARN] ", false);
    System.out.println(learnResult);

    return joined;
  }

  private static void eval(final Trans ensemble, final Pool<?> test) {
    Interval.start();
    final MultiClassModel multiclassModel = new MultiClassModel(ensemble);
    final String testResult = MCTools.evalModel(multiclassModel, test, "[TEST] ", false);
    System.out.println(testResult);
    Interval.stopAndPrint(" evaluation on test");
  }

  public static void main(String[] args) throws IOException, ParseException {
    CommandLineParser parser = new DefaultParser();
    try {
      CommandLine cmd = parser.parse(options, args);

      final String trainPath = cmd.getOptionValue("train");
      final String testPath = cmd.getOptionValue("test");
      final int iterCount = Integer.parseInt(cmd.getOptionValue("iter_count"));
      final double step = Double.parseDouble(cmd.getOptionValue("step"));

      final double gamma = Double.parseDouble(cmd.getOptionValue("gamma", "100"));
      final int maxIter = Integer.parseInt(cmd.getOptionValue("max_iter", "2000"));
      final int depth = Integer.parseInt(cmd.getOptionValue("depth", "5"));
      final int binFactor = Integer.parseInt(cmd.getOptionValue("bin_factor", "32"));

      final FastRandom rng = new FastRandom(0);

      final InputStreamReader trainReader = new InputStreamReader(new GZIPInputStream(new FileInputStream(trainPath)));
      final Pool<?> train = DataTools.loadFromFeaturesTxt(trainPath, trainReader);

      final InputStreamReader testReader = new InputStreamReader(new GZIPInputStream(new FileInputStream(testPath)));
      final Pool<?> test = DataTools.loadFromFeaturesTxt(testPath, testReader);

      final FMCBoosting boosting = new FMCBoosting(
              new StochasticALS(rng, gamma, maxIter, new StochasticALS.Cache(600, 0.01, rng)),
              new GreedyObliviousTree<L2>(GridTools.medianGrid(train.vecData(), binFactor), depth),
              L2.class,
              iterCount,
              step
      );

      final Trans ensemble = fit(boosting, train, test);
      eval(ensemble, test);
    } catch (Exception e) {
      throw e;
     // System.err.println(e);
    }
  }
}
