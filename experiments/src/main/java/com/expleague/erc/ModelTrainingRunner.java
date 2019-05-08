package com.expleague.erc;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.data.*;
import com.expleague.erc.lambda.*;
import com.expleague.erc.metrics.MetricsWriter;
import com.expleague.erc.models.Model;
import com.expleague.erc.models.ModelDays;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

public class ModelTrainingRunner {
    private static final String FILE_MODEL = "model";
    private static final String FILE_USER_MAP = "users_by_id.txt";
    private static final String FILE_ITEM_MAP = "items_by_id.txt";
    private static final String FILE_PREDICTION = "prediction.txt";
    private static final String FILE_SESSIONS = "sessions.txt";
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
        options.addOption(Option.builder().longOpt("toloka").desc("Read data in Toloka format").hasArg(false).build());
    }

    public static void main(String... args) throws ParseException, IOException, ClassNotFoundException {
        final CommandLineParser parser = new DefaultParser();
        final CommandLine cliOptions = parser.parse(options, args);

        final String dataPath = cliOptions.getOptionValue("ds", "../erc/data/lastfm/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv");
        int dim = Integer.parseInt(cliOptions.getOptionValue("dm", "15"));
        double beta = Double.parseDouble(cliOptions.getOptionValue("b", "1e-1"));
        double otherItemImportance = Double.parseDouble(cliOptions.getOptionValue("o", "1e-1"));
        double eps = Double.parseDouble(cliOptions.getOptionValue("e", "0.5"));
        int size = Integer.parseInt(cliOptions.getOptionValue("s", "1000000"));
        int usersNum = Integer.parseInt(cliOptions.getOptionValue("un", "1000"));
        int itemsNum = Integer.parseInt(cliOptions.getOptionValue("in", "1000"));
        double trainRatio = Double.parseDouble(cliOptions.getOptionValue("tr", "0.75"));
        boolean isTop = Boolean.parseBoolean(cliOptions.getOptionValue("t", "true"));
        int iterations = Integer.parseInt(cliOptions.getOptionValue("it", "35"));
        double lr = Double.parseDouble(cliOptions.getOptionValue("lr", "1e-3"));
        double lrd = Double.parseDouble(cliOptions.getOptionValue("lrd", "1"));
        String modelName = cliOptions.getOptionValue("mn", "experiments/src/main/resources/com/expleague/erc/models/model");
        boolean reset = cliOptions.hasOption("r");
        boolean toloka = cliOptions.hasOption("toloka");

        BaseDataReader dataReader = toloka ? new TolokaDataReader() : new LastFmDataReader();
        List<Event> history = dataReader.readData(dataPath, size);
        Map<Integer, String> itemIdToName = dataReader.getReversedItemMap();
        Map<Integer, String> userIdToName = dataReader.getReversedUserMap();
        runModel(history, iterations, lr, lrd, dim, beta, otherItemImportance, eps, usersNum, itemsNum, trainRatio,
                isTop, modelName, itemIdToName, userIdToName, reset);
    }

    private static void saveSessions(Path modelDirPath, List<Event> events) throws IOException {
        final Path sessionsPath = modelDirPath.resolve(FILE_SESSIONS);
        final String sessionsTest = DataPreprocessor.groupEventsToSessions(events).stream()
                .map(session -> session.userId() + "\t" + session.getStartTs() + "\t" + session.getDelta())
                .collect(Collectors.joining("\n", "", "\n"));
        Files.write(sessionsPath, sessionsTest.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    private static void runModel(final List<Event> history, final int iterations, final double lr, final double decay,
                                 final int dim, final double beta, final double otherItemImportance, final double eps,
                                 final int usersNum, final int itemsNum, final double trainRatio, final boolean isTop,
                                 final String modelName, final Map<Integer, String> itemIdToName,
                                 final Map<Integer, String> userIdToName, boolean reset) throws IOException, ClassNotFoundException {
        DataPreprocessor preprocessor = new OneTimeDataProcessor();
        DataPreprocessor.TrainTest dataset = preprocessor.splitTrainTest(history, trainRatio);
        dataset = preprocessor.filter(dataset, usersNum, itemsNum, isTop);
        dataset = preprocessor.filterComparable(dataset);
        final List<Event> train = dataset.getTrain();
        final List<Event> test = dataset.getTest();

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
            LambdaStrategyFactory perUserLambdaStrategyFactory =
                    new PerUserLambdaStrategy.Factory(UserLambdaSingle.makeUserLambdaInitialValues(train));

//            final Model innerDayModel = new ModelUserK(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
//                    new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory());
            model = new ModelDays(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
                    perUserLambdaStrategyFactory);
        }

        saveSessions(modelDirPath, history);

        final MetricsWriter metricsWriter = new MetricsWriter(train, test, eps, modelDirPath);
        model.fit(test, lr, iterations, decay, metricsWriter);

//        model.write(Files.newOutputStream(modelPath));
    }

}
