package com.expleague.erc;

import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.data.LastFmDataReader;
import com.expleague.erc.data.OneTimeDataProcessor;
import com.expleague.erc.lambda.NotLookAheadLambdaStrategy;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.stream.Collectors;

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
    }

    public static void main(String... args) throws ParseException, IOException {
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

        LastFmDataReader lastFmDataReader = new LastFmDataReader();
        List<Event> data = lastFmDataReader.readData(dataPath, size);
        Map<String, Integer> itemNameToId = lastFmDataReader.getItemMap();
        Map<Integer, String> itemIdToName = itemNameToId.keySet().stream()
                .collect(Collectors.toMap(itemNameToId::get, Function.identity()));
        runModel(data, iterations, lr, lrd, dim, beta, otherItemImportance, eps, usersNum, itemsNum, trainRatio, isTop,
                spuLogPath, itemIdToName);
    }

    private static void runModel(final List<Event> data, final int iterations, final double lr, final double decay,
                                 final int dim, final double beta, final double otherItemImportance, final double eps,
                                 final int usersNum, final int itemsNum, final double trainRatio, final boolean isTop,
                                 final String spuLogFilePath, final Map<Integer, String> itemIdToName) {
        DataPreprocessor preprocessor = new OneTimeDataProcessor();
        DataPreprocessor.TrainTest dataset = preprocessor.splitTrainTest(data, trainRatio);
        dataset = preprocessor.filter(dataset, usersNum, itemsNum, isTop);
//        DoubleUnaryOperator lambdaTransform = Math::abs;
//        DoubleUnaryOperator lambdaDerivative = Math::signum;
        DoubleUnaryOperator lambdaTransform = x -> x;
        DoubleUnaryOperator lambdaDerivative = x -> 1;
        Model model = new Model(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative, new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory());
        model.initializeEmbeddings(dataset.getTrain());
        MetricsCalculator metricsCalculator = null;
        final Path spuLogPath = spuLogFilePath != null ? Paths.get(spuLogFilePath) : null;
        try {
            metricsCalculator = new MetricsCalculator(dataset.getTrain(), dataset.getTest(), spuLogPath, itemIdToName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Constant prediction: " + metricsCalculator.constantPredictionTimeMae());
        System.out.println("Target mean SPU: " + metricsCalculator.getMeanSpuTarget());
        metricsCalculator.printMetrics(model);
        model.fit(dataset.getTrain(), lr, iterations, dataset.getTest(), decay, true, metricsCalculator);
    }
}
