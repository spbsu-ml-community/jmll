package com.expleague.fmcb;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.GridTools;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.multiclass.gradfac.FMCBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.MultiClassModel;
import org.apache.commons.cli.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
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
            .longOpt("valid")
            .desc("Path to the valid dataset")
            .hasArg()
            .argName("VALID")
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
//    options.addOption(Option.builder()
//            .longOpt("early_stopping_rounds")
//            .desc("Early stopping rounds")
//            .hasArg()
//            .argName("EARLY_STOPPING_ROUNDS")
//            .type(Integer.class)
//            .build());
    options.addOption(Option.builder()
            .longOpt("ensemble_size")
            .desc("Ensemble size")
            .hasArg()
            .argName("ENSEMBLE_SIZE")
            .type(Integer.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("n_iter")
            .desc("Number of weak learners")
            .hasArg()
            .argName("N_ITER")
            .type(Integer.class)
            .type(Number.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("step")
            .desc("Learning rate")
            .hasArg()
            .argName("STEP")
            .type(Number.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("lambda")
            .desc("L1 loss coefficient")
            .hasArg()
            .argName("LAMBDA")
            .type(Number.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("train_pred")
            .desc("Path to the file with predictions on training dataset")
            .hasArg()
            .argName("TRAIN_PRED")
            .type(String.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("valid_pred")
            .desc("Path to the file with predictions on valid dataset")
            .hasArg()
            .argName("VALID_PRED")
            .type(String.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("test_pred")
            .desc("Path to the file with predictions on test dataset")
            .hasArg()
            .argName("TEST_PRED")
            .type(String.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("is_gbdt")
            .desc("Gradient boosting or random forest")
            .hasArg()
            .argName("IS_GBDT")
            .type(Boolean.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("format")
            .desc("Dataset format")
            .hasArg()
            .argName("FORMAT")
            .type(String.class)
            .build());
  }

  private static Trans fit(final FMCBoosting boosting, final Pool<?> train, final Pool<?> valid) {
    final boolean isMultiLabel = train.tcount() > 1;
    FMCBoosting.Metric metric = isMultiLabel ? new FMCBoosting.Metric.Precision(5) : new FMCBoosting.Metric.Accuracy();

    final List<Consumer<Trans>> evaluators = new ArrayList<>();

    evaluators.add(new Consumer<Trans>() {
      private static final int INTERVAL = 100;
      int iteration;

      @Override
      public void accept(Trans trans) {
        if (++iteration % INTERVAL == 0) {
          System.out.println("\nIteration #" + iteration);
        }
      }
    });

    evaluators.add(new FMCBoosting.Evaluator("Train", train, metric));

    if (valid != null) {
      evaluators.add(new FMCBoosting.Evaluator("Valid", valid, metric));
    }

    // Add listeners
    evaluators.forEach(boosting::addListener);

    long startTime = System.currentTimeMillis();
    final Ensemble ensemble = boosting.fit(train);
    System.out.println("\nTraining time: " + (System.currentTimeMillis() - startTime) + "(ms)");

    final Trans joined = ensemble.last() instanceof FuncJoin ? MCTools.joinBoostingResult(ensemble) : ensemble;

    return joined;
  }

  private static Vec predictClass(final Trans ensemble, final Pool<?> pool) {
    final MultiClassModel multiclassModel = new MultiClassModel(ensemble);
    return multiclassModel.bestClassAll(pool.vecData().data(), true);
  }

  private static Mx predictScores(final Trans ensemble, final Pool<?> pool) {
    final Mx prediction = ensemble.transAll(pool.vecData().data(), true);
    final Mx score = new VecBasedMx(prediction.rows(), prediction.columns() + 1);

    // Copy scores
    for (int i = 0; i < prediction.columns(); ++i) {
      VecTools.assign(score.col(i), prediction.col(i));
    }

    return score;
  }

  private static void saveIntVec(final Vec data, final String path) throws Exception {
    final String result = data.stream().mapToObj(Math::round).map(Object::toString).collect(Collectors.joining(","));
    final PrintStream out = new PrintStream(new FileOutputStream(new File(path)));
    out.println(result);
    out.close();
  }

  private static void saveMx(final Mx data, final String path) throws Exception {
    final PrintStream out = new PrintStream(new FileOutputStream(new File(path)));

    for (int i = 0; i < data.rows(); ++i) {
      final DoubleStream values = data.row(i).stream();
      final String row = values.mapToObj(Double::toString).collect(Collectors.joining(","));
      out.println(row);
    }

    out.close();
  }

  private static Pool<?> loadPool(String fileName, String format) throws IOException {
    final FileInputStream file = new FileInputStream(fileName);
    final InputStream in = fileName.endsWith("gz") ? new GZIPInputStream(file) : file;
    InputStreamReader reader = new InputStreamReader(in);

    if (format.equals("xml")) {
      return DataTools.loadFromXMLFormat(fileName, reader);
    }

    return DataTools.loadFromFeaturesTxt(fileName, reader);
  }

  private static void evaluateAndSave(final Trans ensemble, final Pool<?> pool, final String path) throws Exception {
    final boolean isMultiLabel = pool.tcount() > 1;

    if (isMultiLabel) {
      final Mx scores = predictScores(ensemble, pool);
      saveMx(scores, path);
    } else {
      final Vec pred = predictClass(ensemble, pool);
      saveIntVec(pred, path);
    }
  }

  public static void main(String[] args) throws Exception {
    CommandLineParser parser = new DefaultParser();
    try {
      CommandLine cmd = parser.parse(options, args);

      final String model = cmd.getOptionValue("model", null);
      final String trainPath = cmd.getOptionValue("train", null);
      final String validPath = cmd.getOptionValue("valid", null);
      final String testPath = cmd.getOptionValue("test", null);

      if (trainPath != null && !cmd.hasOption("n_iter")) {
        throw new IllegalArgumentException("You should specify iterations count!");
      }

      if (trainPath != null && !cmd.hasOption("step")) {
        throw new IllegalArgumentException("You should specify step!");
      }

      final int iterCount = Integer.parseInt(cmd.getOptionValue("n_iter", "1000"));
      final double step = Double.parseDouble(cmd.getOptionValue("step", "1"));
      final double gamma = Double.parseDouble(cmd.getOptionValue("gamma", "100"));
      final double lambda = Double.parseDouble(cmd.getOptionValue("lambda", "0"));
      final int maxIter = Integer.parseInt(cmd.getOptionValue("max_iter", "1000"));
      final int depth = Integer.parseInt(cmd.getOptionValue("depth", "5"));
      // final int earlyStoppingRounds = Integer.parseInt(cmd.getOptionValue("early_stopping_rounds", "0"));
      final int binFactor = Integer.parseInt(cmd.getOptionValue("n_bins", "32"));
      final String trainPredPath = cmd.getOptionValue("train_pred", null);
      final String validPredPath = cmd.getOptionValue("valid_pred", null);
      final String testPredPath = cmd.getOptionValue("test_pred", null);
      final int ensembleSize = Integer.parseInt(cmd.getOptionValue("ensemble_size", "5"));
      final boolean isGbdt = Boolean.parseBoolean(cmd.getOptionValue("is_gbdt", "true"));
      final String format = cmd.getOptionValue("format", "txt");
      final FastRandom rng = new FastRandom(0);

      Pool<?> train = trainPath != null ? loadPool(trainPath, format) : null;
      Pool<?> valid = validPath != null ? loadPool(validPath, format) : null;
      Pool<?> test = testPath != null ? loadPool(testPath, format) : null;

      Trans ensemble = null;
      if (train == null && model != null) {
        ensemble = DataTools.deprecatedReadModel(new FileInputStream(new File(model)));
      }

      if (train == null && test == null) {
        throw new IllegalArgumentException("Either train or test dataset is required!");
      }

      if (train == null && ensemble == null) {
        throw new IllegalArgumentException("You should specify the model for test evaluation!");
      }

      if (train != null) {
        final FMCBoosting boosting = new FMCBoosting(
                new StochasticALS(rng, gamma, maxIter, lambda, 0.0, null),
                new GreedyObliviousTree<StatBasedLoss>(GridTools.medianGrid(train.vecData(), binFactor), depth),
                L2.class,
                iterCount,
                step,
                ensembleSize,
                isGbdt
        );

        ensemble = fit(boosting, train, valid);

        if (model != null) {
          DataTools.writeModel(ensemble, new FileOutputStream(new File(model)));
        }
      }

      if (train != null && trainPredPath != null) {
        evaluateAndSave(ensemble, train, trainPredPath);
      }

      if (valid != null && validPredPath != null) {
        evaluateAndSave(ensemble, valid, validPredPath);
      }

      if (test != null && testPredPath != null) {
        evaluateAndSave(ensemble, test, testPredPath);
      }
    } catch (Exception e) {
      throw e;
    }
  }
}
