package com.expleague.erc;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.lambda.*;
import com.expleague.erc.metrics.MetricsWriter;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.data.LastFmDataReader;
import com.expleague.erc.data.OneTimeDataProcessor;
import com.expleague.erc.models.Model;
import com.expleague.erc.models.ModelPerUser;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;

public class ModelTrainingRunner {
    private static final String FILE_MODEL = "model";
    private static final String FILE_USER_MAP = "users_by_id.txt";
    private static final String FILE_ITEM_MAP = "items_by_id.txt";
    private static final String FILE_PREDICTION = "prediction.txt";
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
        options.addOption(Option.builder("mn").longOpt("model_name").desc("Name for statistics files").hasArg().build());
        options.addOption(Option.builder("r").longOpt("reset").desc("Wipe the model directory").hasArg(false).build());
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
        String modelName = cliOptions.getOptionValue("mn", null);
        boolean reset = cliOptions.hasOption("r");

        LastFmDataReader lastFmDataReader = new LastFmDataReader();
        List<Event> data = lastFmDataReader.readData(dataPath, size);
        Map<Integer, String> itemIdToName = lastFmDataReader.getReversedItemMap();
        Map<Integer, String> userIdToName = lastFmDataReader.getReversedUserMap();
        runModel(data, iterations, lr, lrd, dim, beta, otherItemImportance, eps, usersNum, itemsNum, trainRatio, isTop,
                modelName, itemIdToName, userIdToName, reset);
    }

    private static void runModel(final List<Event> data, final int iterations, final double lr, final double decay,
                                 final int dim, final double beta, final double otherItemImportance, final double eps,
                                 final int usersNum, final int itemsNum, final double trainRatio, final boolean isTop,
                                 final String modelName, final Map<Integer, String> itemIdToName,
                                 final Map<Integer, String> userIdToName, boolean reset) throws IOException, ClassNotFoundException {
        DataPreprocessor preprocessor = new OneTimeDataProcessor();
        DataPreprocessor.TrainTest dataset = preprocessor.splitTrainTest(preprocessor.filterSessions(data), trainRatio);
        dataset = preprocessor.filter(dataset, usersNum, itemsNum, isTop);
        dataset = preprocessor.filterComparable(dataset);

        Path modelDirPath = Paths.get(modelName);
        boolean existingModel = Files.isDirectory(modelDirPath);

        final Model model;

        final Path modelPath = modelDirPath.resolve(FILE_MODEL);
        if (existingModel && !reset) {
            model = Model.load(Files.newInputStream(modelPath));
        } else {
            if (!existingModel) {
                Files.createDirectory(modelDirPath);
            }
            if (reset) {
                Files.list(modelDirPath).forEach(path -> {
                    try {
                        Files.delete(path);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
            }
            Util.writeMap(modelDirPath.resolve(FILE_USER_MAP), userIdToName);
            Util.writeMap(modelDirPath.resolve(FILE_ITEM_MAP), itemIdToName);

            DoubleUnaryOperator lambdaTransform = new LambdaTransforms.AbsTransform();
            DoubleUnaryOperator lambdaDerivative = new LambdaTransforms.AbsDerivativeTransform();
            TIntObjectMap<Vec> userEmbeddings = new TIntObjectHashMap<>();
            TIntObjectMap<Vec> itemEmbeddings = new TIntObjectHashMap<>();
            Model.makeInitialEmbeddings(dim, dataset.getTrain(), userEmbeddings, itemEmbeddings);
            LambdaStrategyFactory perUserLambdaStrategyFactory =
                    new PerUserLambdaStrategy.Factory(UserLambdaSingle.makeUserLambdaInitialValues(dataset.getTrain()));
            model = new ModelPerUser(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
                    perUserLambdaStrategyFactory, userEmbeddings, itemEmbeddings);
//            TIntDoubleMap userBaseLambdas = new TIntDoubleHashMap();
//            TIntIntMap userKs = new TIntIntHashMap();
//            ModelUserK.calcUserParams(dataset.getTrain(), userBaseLambdas, userKs);
//            ModelUserK.makeInitialEmbeddings(dim, userBaseLambdas, dataset.getTrain(), userEmbeddings, itemEmbeddings);
//            model = new ModelUserK(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
//                    new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory(), userEmbeddings, itemEmbeddings,
//                    userKs, userBaseLambdas);

        }

        final MetricsWriter metricsWriter =
                new MetricsWriter(dataset.getTrain(), dataset.getTest(), eps, modelDirPath);
        model.fit(dataset.getTrain(), lr, iterations, decay, metricsWriter);

        model.write(Files.newOutputStream(modelPath));
    }

}
