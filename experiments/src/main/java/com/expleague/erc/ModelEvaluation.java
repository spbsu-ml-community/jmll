package com.expleague.erc;

import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.data.LastFmDataReader;
import com.expleague.erc.data.OneTimeDataProcessor;
import com.expleague.erc.lambda.NotLookAheadLambdaStrategy;
import com.expleague.erc.lambda.UserLambda;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.function.DoubleUnaryOperator;

public class ModelEvaluation {
    private static Options options = new Options();
    static {
        options.addOption(Option.builder("ds").longOpt("dataset").desc("Path to data").hasArg().build());
        options.addOption(Option.builder("it").longOpt("iter").desc("Num of iterations").hasArg().build());
        options.addOption(Option.builder("lr").longOpt("learning_rate").desc("Learning rate").hasArg().build());
        options.addOption(Option.builder("lrd").longOpt("learning_rate_decay").desc("Learning rate decay").hasArg().build());
        options.addOption(Option.builder("dm").longOpt("dim").desc("Dimension of embeddings").hasArg().build());
        options.addOption(Option.builder("b").longOpt("beta").desc("Beta").hasArg().build());
        options.addOption(Option.builder("o").longOpt("other_items_importance").desc("Other items importance").hasArg().build());
        options.addOption(Option.builder("e").longOpt("eps").desc("Epsilon").hasArg().build());
        options.addOption(Option.builder("s").longOpt("size").desc("Num of lines read from data").hasArg().build());
        options.addOption(Option.builder("tr").longOpt("train_ratio").desc("Train data ratio to all data size").hasArg().build());
        options.addOption(Option.builder("un").longOpt("user_num").desc("Num of users").hasArg().build());
        options.addOption(Option.builder("in").longOpt("item_num").desc("Num of items").hasArg().build());
        options.addOption(Option.builder("t").longOpt("top").desc("Is filter on top items").hasArg().build());
        options.addOption(Option.builder("spup").longOpt("spupath").desc("Path to save item SPUs").hasArg().build());
        options.addOption(Option.builder("ml").longOpt("model_load").desc("Path to load model from").hasArg().build());
        options.addOption(Option.builder("ms").longOpt("model_save").desc("Path to save model").hasArg().build());
        options.addOption(Option.builder("ump").longOpt("user_map_path").desc("Path to save users map").hasArg().build());
        options.addOption(Option.builder("imp").longOpt("item_map_path").desc("Path to save items map").hasArg().build());
        options.addOption(Option.builder("lp").longOpt("lambdas_path").desc("Path to save lambdas").hasArg().build());
    }

    public static void main(String... args) throws ParseException, IOException, ClassNotFoundException {
        final CommandLineParser parser = new DefaultParser();
        final CommandLine cliOptions = parser.parse(options, args);

        final String dataPath = cliOptions.getOptionValue("ds", "~/data/mlimlab/erc/datasets/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname_1M.tsv");
        int dim = Integer.parseInt(cliOptions.getOptionValue("dm", "15"));
        double beta = Double.parseDouble(cliOptions.getOptionValue("b", "1e-1"));
        double otherItemImportance = Double.parseDouble(cliOptions.getOptionValue("o", "1e-1"));
        double eps = Double.parseDouble(cliOptions.getOptionValue("e", "5"));
        int size = Integer.parseInt(cliOptions.getOptionValue("s", "1000000"));
        int usersNum = Integer.parseInt(cliOptions.getOptionValue("un", "1000"));
        int itemsNum = Integer.parseInt(cliOptions.getOptionValue("in", "1000"));
        double trainRatio = Double.parseDouble(cliOptions.getOptionValue("tr", "0.75"));
        boolean isTop = Boolean.parseBoolean(cliOptions.getOptionValue("t", "true"));
        int iterations = Integer.parseInt(cliOptions.getOptionValue("it", "15"));
        double lr = Double.parseDouble(cliOptions.getOptionValue("lr", "1e-3"));
        double lrd = Double.parseDouble(cliOptions.getOptionValue("lrd", "1"));
        String spuLogPath = cliOptions.getOptionValue("spup", null);
        String modelLoadPath = cliOptions.getOptionValue("ml", null);
        String modelSavePath = cliOptions.getOptionValue("ms", null);
        String usersMapPath = cliOptions.getOptionValue("ump", null);
        String itemsMapPath = cliOptions.getOptionValue("imp", null);
        String lambdasPath = cliOptions.getOptionValue("lp", null);

        LastFmDataReader lastFmDataReader = new LastFmDataReader();
        List<Event> data = lastFmDataReader.readData(dataPath, size);
        Map<Integer, String> itemIdToName = lastFmDataReader.getReversedItemMap();
        Map<Integer, String> userIdToName = lastFmDataReader.getReversedUserMap();
        runModel(data, iterations, lr, lrd, dim, beta, otherItemImportance, eps, usersNum, itemsNum, trainRatio, isTop,
                spuLogPath, itemIdToName, userIdToName, modelLoadPath, modelSavePath, usersMapPath, itemsMapPath,
                lambdasPath);
    }

    private static void runModel(final List<Event> data, final int iterations, final double lr, final double decay,
                                 final int dim, final double beta, final double otherItemImportance, final double eps,
                                 final int usersNum, final int itemsNum, final double trainRatio, final boolean isTop,
                                 final String spuLogFilePath, final Map<Integer, String> itemIdToName,
                                 final Map<Integer, String> userIdToName, final String modelLoadPath,
                                 final String modelSavePath, final String usersMapPath, final String itemsMapPath,
                                 final String lambdasPath) throws IOException, ClassNotFoundException {
        DataPreprocessor preprocessor = new OneTimeDataProcessor();
        DataPreprocessor.TrainTest dataset = preprocessor.splitTrainTest(preprocessor.filterSessions(data), trainRatio);
        dataset = preprocessor.filter(dataset, usersNum, itemsNum, isTop);
        dataset = preprocessor.filterComparable(dataset);

        if (usersMapPath != null) {
            Util.writeMap(Paths.get(usersMapPath), userIdToName);
        }
        if (itemsMapPath != null) {
            Util.writeMap(Paths.get(itemsMapPath), itemIdToName);
        }

        Model model;
        if (modelLoadPath == null) {
            DoubleUnaryOperator lambdaTransform = new UserLambda.IdentityTransform();
            DoubleUnaryOperator lambdaDerivative = new UserLambda.IdentityDerivativeTransform();
            model = new Model(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
                    new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory());
        } else {
            model = Model.load(Files.newInputStream(Paths.get(modelLoadPath)));
        }

        model.initializeEmbeddings(dataset.getTrain());
        final Path spuLogPath = spuLogFilePath != null ? Paths.get(spuLogFilePath) : null;
        final MetricsCalculator metricsCalculator = new MetricsCalculator(dataset.getTrain(), dataset.getTest(), spuLogPath);
        if (modelLoadPath == null) {
            metricsCalculator.writePairNames(spuLogPath, itemIdToName, userIdToName);
            metricsCalculator.writeTargetSpus();
        }
        System.out.println("Constant prediction: " + metricsCalculator.constantPredictionTimeMae());
        System.out.println("Target mean SPU: " + metricsCalculator.getMeanSpuTarget());
        try {
            final MetricsCalculator.Summary summary = metricsCalculator.calculateSummary(model);
            System.out.println(summary);
            if (modelLoadPath == null) {
                summary.writeSpus();
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        model.fit(dataset.getTrain(), lr, iterations, dataset.getTest(), decay, true, metricsCalculator, modelSavePath);

        if (modelSavePath != null) {
            model.write(Files.newOutputStream(Paths.get(modelSavePath)));
        }
        if (lambdasPath != null) {
            metricsCalculator.writeLambdas(Paths.get(lambdasPath), itemIdToName, userIdToName, model);
        }
    }
}
