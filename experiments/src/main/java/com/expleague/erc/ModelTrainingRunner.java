package com.expleague.erc;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.data.*;
import com.expleague.erc.lambda.*;
import com.expleague.erc.metrics.*;
import com.expleague.erc.models.*;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

public class ModelTrainingRunner {
    private static final String FILE_MODEL = "model";
    private static final String FILE_USER_MAP = "users_by_id.txt";
    private static final String FILE_ITEM_MAP = "items_by_id.txt";
    private static final String FILE_PREDICTION = "prediction.txt";
    private static final String FILE_SESSIONS = "sessions.txt";
    private static final String FILE_ITEM_EMBEDDINGS = "item_embeddings.txt";
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
        final List<Event> train = dataset.getTrain();
        final List<Event> test = dataset.getTest();

        Path modelDirPath = Paths.get(modelName);
        boolean existingModel = Files.isDirectory(modelDirPath);

        final ModelCombined model;

        final Path modelPath = modelDirPath.resolve(FILE_MODEL);
        if (existingModel && !reset) {
            model = ModelCombined.load(modelPath);
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
            LambdaStrategyFactory perUserLambdaStrategyFactory = new PerUserLambdaStrategy.Factory();

//            final Model innerDayModel = new ModelUserK(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
//                    new NotLookAheadLambdaStrategy.NotLookAheadLambdaStrategyFactory());
//            model = new ModelDays(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
//                    perUserLambdaStrategyFactory);
            model = new ModelCombined(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
                    perUserLambdaStrategyFactory);
//            model = new ModelExpPerUser(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivative,
//                    new PerUserLambdaStrategy.Factory(), new Util.GetDelta(), Double.NEGATIVE_INFINITY, DAY_HOURS);
        }

        saveSessions(modelDirPath, dataset.getTest());

        evaluateConstant(train, test);
        final MetricsWriter combinedWriter =
                new MetricsWriter(train, test, null, new MAEPerUser(), new SPUPerUser(), false);
        final MetricsWriter dailyWriter = new MetricsWriter(train, test, new LogLikelihoodDaily(train),
                new MAEDaily(train), null, true);
        model.setFitListener(combinedWriter);
        model.setDaysFitListener(dailyWriter);
        model.fit(train, lr, iterations, decay);

        model.save(modelPath);
        writeEmbeddings(model, modelDirPath);
    }

    private static TIntDoubleMap calcConstants(List<Event> data, ToDoubleFunction<Session> getTime) {
        final TIntObjectMap<TDoubleList> userDeltas = new TIntObjectHashMap<>();
        for (final Session session : DataPreprocessor.groupEventsToSessions(data)) {
            final int userId = session.userId();
            if (Util.forPrediction(session)) {
                if (!userDeltas.containsKey(userId)) {
                    userDeltas.put(userId, new TDoubleArrayList());
                }
                userDeltas.get(userId).add(getTime.applyAsDouble(session));
            }
        }
        final TIntDoubleMap constants = new TIntDoubleHashMap();
        userDeltas.forEachEntry((userId, deltas) -> {
            deltas.sort();
            constants.put(userId, deltas.get(deltas.size() / 2));
            return true;
        });
        return constants;
    }

    private static double calcDefaultConstant(List<Event> data, ToDoubleFunction<Session> getTime) {
        final double[] intervals = DataPreprocessor.groupEventsToSessions(data).stream()
                .filter(Util::forPrediction)
                .mapToDouble(getTime)
                .sorted().toArray();
        return intervals[intervals.length / 2];
    }

    private static class ConstantApplicable implements ApplicableModel {
        private final TIntDoubleMap constants;
        private final double defaultConstant;

        private ConstantApplicable(TIntDoubleMap constants, double justConstant) {
            this.constants = constants;
            this.defaultConstant = justConstant;
        }

        @Override
        public void accept(EventSeq event) {}

        @Override
        public double timeDelta(final int userId, final double time) {
            if (constants.containsKey(userId)) {
                return constants.get(userId);
            } else {
                return defaultConstant;
            }
        }
    }

    private static void evaluateConstant(final List<Event> trainData, final List<Event> testData) {
        final TIntIntMap userDayBorders = ModelCombined.findMinHourInDay(trainData);

        final TIntDoubleMap dayConstants = calcConstants(trainData,
                session -> Util.getDaysFromPrevSession(session, userDayBorders.get(session.userId())));
        final double dayDefaultConstant = calcDefaultConstant(trainData,
                session -> Util.getDaysFromPrevSession(session, userDayBorders.get(session.userId())));
        final TIntDoubleMap hourConstants = calcConstants(trainData, Session::getDelta);
        final double hourDefaultConstant = calcDefaultConstant(trainData, Session::getDelta);

        final ApplicableModel dayApplicable = new ConstantApplicable(dayConstants, dayDefaultConstant);
        final ApplicableModel hourApplicable = new ConstantApplicable(hourConstants, hourDefaultConstant);

        final Metric maeDays = new MAEDaily(trainData);
        final Metric maeHours = new MAEPerUser();
        final Metric spu = new SPUPerUser();

        final ForkJoinTask<Double> trainMaeDaysTask = ForkJoinPool.commonPool().submit(() ->
                maeDays.calculate(trainData, dayApplicable));
        final ForkJoinTask<Double> testMaeDaysTask = ForkJoinPool.commonPool().submit(() ->
                maeDays.calculate(testData, dayApplicable));
        final ForkJoinTask<Double> trainMaeHoursTask = ForkJoinPool.commonPool().submit(() ->
                maeHours.calculate(trainData, hourApplicable));
        final ForkJoinTask<Double> testMaeHoursTask = ForkJoinPool.commonPool().submit(() ->
                maeHours.calculate(testData, hourApplicable));
        final ForkJoinTask<Double> trainSpuTask = ForkJoinPool.commonPool().submit(() ->
                spu.calculate(trainData, hourApplicable));
        final ForkJoinTask<Double> testSpuTask = ForkJoinPool.commonPool().submit(() ->
                spu.calculate(testData, hourApplicable));

        try {
            System.out.printf("train_const_days_mae: %f, test_const_days_mae: %f\n",
                    trainMaeDaysTask.get(), testMaeDaysTask.get());
            System.out.printf("train_const_hours_mae: %f, test_const_hours_mae: %f, train_const_hours_spu: %f, " +
                            "test_const_hours_spu: %f\n\n",
                    trainMaeHoursTask.get(), testMaeHoursTask.get(), trainSpuTask.get(), testSpuTask.get());
        } catch (InterruptedException | ExecutionException e) {
            System.out.println("Constant evaluation failed: " + e.getMessage());
        }
    }

    private static void writeEmbeddings(Model model, Path modelDirPath) throws IOException {
        final TIntObjectMap<Vec> itemEmbeddings = model.getItemEmbeddings();
        final String embeddingsStr = Files.lines(modelDirPath.resolve(FILE_ITEM_MAP))
                .map(line -> {
                    final String[] desc = line.split("\t");
                    final int number = Integer.parseInt(desc[0]);
                    final String name = desc[1];
                    final Vec embedding = itemEmbeddings.get(number);
                    if (embedding == null) {
                        return null;
                    }
                    return name + "\t" + embedding.stream()
                            .mapToObj(String::valueOf)
                            .collect(Collectors.joining(" "));
                })
                .filter(Objects::nonNull)
                .collect(Collectors.joining("\n"));

        Files.write(modelDirPath.resolve(FILE_ITEM_EMBEDDINGS), embeddingsStr.getBytes());
    }
}
