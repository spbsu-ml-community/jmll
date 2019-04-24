package com.expleague.erc;

import com.expleague.erc.models.Model;
import gnu.trove.iterator.TIntIterator;
import gnu.trove.iterator.TLongIntIterator;
import gnu.trove.map.*;
import gnu.trove.map.hash.*;
import gnu.trove.set.TIntSet;
import gnu.trove.set.TLongSet;
import gnu.trove.set.hash.TIntHashSet;
import gnu.trove.set.hash.TLongHashSet;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.lang.Math;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.stream.Collectors;

public class MetricsCalculator {
    private static final double DAY = 24;
    private static final double EPS = .1;
    private static final String FILE_SPU_TRAIN = "spu_train.txt";
    private static final String FILE_SPU_TEST = "spu_test.txt";
    private static final String FILE_LAMBDAS_INITIAL = "init_lambdas.txt";
    private static final String FILE_LAMBDAS_TRAINED = "trained_lambdas.txt";

    private final List<Event> trainData;
    private final List<Event> testData;
    private final int[] itemIds;
    private final double startTime;
    private final double splitTime;
    private final double endTime;
    private final TLongDoubleMap targetPairwiseSPU;
    private final double targetPairwiseSPUMean;
    private final TIntObjectMap<TIntSet> itemsUsers;
    private final TLongSet relevantPairs;
    private final long[] relevantPairsArray;
    private final TIntObjectMap<int[]> itemsUsersArrays;
    private final TLongDoubleMap lastTrainEvents;
    private final TLongDoubleMap beginningTimes;
    private final ForkJoinPool pool;
    private final Path spuTestPath;
    private final Path spuTrainPath;
    private final Path lambdaInitialPath;
    private final Path lambdaTrainedPath;

    public MetricsCalculator(List<Event> trainData, List<Event> testData, Path saveDir) {
        this.trainData = trainData;
        this.testData = testData;
        startTime = trainData.get(0).getTs();
        splitTime = trainData.get(trainData.size() - 1).getTs();
        endTime = testData.get(testData.size() - 1).getTs();
        itemsUsers = selectUsers(trainData, testData);
        relevantPairs = selectPairs(itemsUsers);
        relevantPairsArray = pairsToArray(relevantPairs);
        this.itemIds = itemsUsers.keys();
        itemsUsersArrays = toArrays(itemsUsers);
        targetPairwiseSPU = pairwiseHistorySpu(testData);
        targetPairwiseSPUMean = Arrays.stream(targetPairwiseSPU.values()).average().orElse(-1);
        spuTrainPath = saveDir.resolve(FILE_SPU_TRAIN);
        spuTestPath = saveDir.resolve(FILE_SPU_TEST);
        lambdaTrainedPath = saveDir.resolve(FILE_LAMBDAS_TRAINED);
        lambdaInitialPath = saveDir.resolve(FILE_LAMBDAS_INITIAL);

        lastTrainEvents = new TLongDoubleHashMap();
        beginningTimes = new TLongDoubleHashMap();
        for (Event event: trainData) {
            long pair = event.getPair();
            lastTrainEvents.put(pair, event.getTs());
            beginningTimes.put(pair, startTime);
        }
        pool = new ForkJoinPool();
    }

    private static TIntObjectMap<TIntSet> selectUsers(final List<Event> trainData, final List<Event> testData) {
        final Map<Integer, Set<Integer>> itemsTrainUsers = trainData.stream()
                .collect(Collectors.groupingBy(Event::itemId, Collectors.mapping(Event::userId, Collectors.toSet())));
        final Map<Integer, Set<Integer>> itemsTestUsers = testData.stream()
                .collect(Collectors.groupingBy(Event::itemId, Collectors.mapping(Event::userId, Collectors.toSet())));
        final TIntObjectMap<TIntSet> itemsUsers = new TIntObjectHashMap<>();
        for (int itemId: itemsTrainUsers.keySet()) {
            if (itemsTestUsers.containsKey(itemId)) {
                final TIntSet curItemUsers = new TIntHashSet(itemsTrainUsers.get(itemId));
                curItemUsers.retainAll(itemsTestUsers.get(itemId));
                if (!curItemUsers.isEmpty()) {
                    itemsUsers.put(itemId, curItemUsers);
                }
            }
        }
        return itemsUsers;
    }

    private static TLongSet selectPairs(TIntObjectMap<TIntSet> itemsUsers) {
        final TLongSet pairs = new TLongHashSet();
        for (int itemId: itemsUsers.keys()) {
            for (TIntIterator it = itemsUsers.get(itemId).iterator(); it.hasNext(); ) {
                int userId = it.next();
                pairs.add(Util.combineIds(userId, itemId));
            }
        }
        return pairs;
    }

    private static TIntObjectMap<int[]> toArrays(TIntObjectMap<TIntSet> itemsUsers) {
        final TIntObjectMap<int[]> itemsUsersArrays = new TIntObjectHashMap<>();
        for (int itemId: itemsUsers.keySet().toArray()) {
            itemsUsersArrays.put(itemId, itemsUsers.get(itemId).toArray());
        }
        return itemsUsersArrays;
    }

    private static long[] pairsToArray(TLongSet pairs) {
        return Arrays.stream(pairs.toArray())
                .boxed()
                .sorted(Comparator.comparingInt(Util::extractItemId).thenComparingInt(Util::extractUserId))
                .mapToLong(Long::longValue)
                .toArray();
    }

    public double returnTimeMae(Model.Applicable model, List<Event> data) {
        double errors = 0.;
        long count = 0;
        for (final Event event : data) {
            count++;
            final double expectedReturnTime = model.timeDelta(event.userId(), event.itemId());
            errors += Math.abs(event.getPrDelta() - expectedReturnTime);
            model.accept(event);
        }
        return errors / count;
    }

    public double itemRecommendationMae(Model model) {
        long errorsSum = 0;
        long count = 0;
        int[] allItemIds = model.getItemEmbeddings().keys();
        Model.Applicable applicable = model.getApplicable(trainData);
        for (Event event: testData) {
            int userId = event.userId();
            final double actualLambda = applicable.getLambda(userId, event.itemId());
            for (int i: allItemIds) {
                if (applicable.getLambda(userId, i) > actualLambda) {
                    errorsSum++;
                }
            }
            count++;
            applicable.accept(event);
        }
        return (double) errorsSum / count;
    }

    public double constantPredictionTimeMae() {
        final double meanItemDelta = trainData.parallelStream()
                .filter(event -> !event.isFinish() && event.getPrDelta() >= 0)
                .collect(Collectors.averagingDouble(Event::getPrDelta));
        return testData.parallelStream()
                .collect(Collectors.averagingDouble(event -> Math.abs(event.getPrDelta() - meanItemDelta)));
    }

    public TLongDoubleMap pairwiseHistorySpu(List<Event> history) {
        final double startTime = history.get(0).getTs();
        final TLongIntMap pairSessions = new TLongIntHashMap();
        final TLongDoubleMap pairDeathTimes = new TLongDoubleHashMap();
        final TLongDoubleMap pairBirthTimes = new TLongDoubleHashMap();
        final TLongDoubleMap pairSpus = new TLongDoubleHashMap();
        for (Event event: history) {
            final long pair = event.getPair();
            if (relevantPairs.contains(pair)) {
                pairDeathTimes.put(pair, event.getTs() - startTime);
                if (!pairBirthTimes.containsKey(pair)) {
//                pairBirthTimes.put(pair, event.getTs() - startTime);
                    pairBirthTimes.put(pair, 0);
                }
            }
        }
        for (Event event: history) {
            final long pair = event.getPair();
            if (relevantPairs.contains(pair)) {
                pairSessions.put(pair, pairSessions.get(pair) + 1);
            }
        }
        for (TLongIntIterator it = pairSessions.iterator(); it.hasNext(); ) {
            it.advance();
            final long pair = it.key();
            final int sessions = it.value();
            pairSpus.put(pair, sessions / ((pairDeathTimes.get(pair) - pairBirthTimes.get(pair)) + DAY - EPS) / DAY);
        }
        return pairSpus;
    }

    private static final int MAX_PREDICTION_LEN = 10000;

    public List<Event> predictSpan(Model.Applicable model, TLongDoubleMap previousActivityTimes,
                                   double spanStartTime, double spanEndTime) {
        final List<Event> generatedEvents = new ArrayList<>();
        final Queue<Event> followingEvents = new ArrayDeque<>();
        for (int itemId: itemIds) {
            for (int userId: itemsUsersArrays.get(itemId)) {
                final double newEventTime = previousActivityTimes.get(Util.combineIds(userId, itemId)) +
                        model.timeDelta(userId, itemId);
                if (newEventTime <= spanEndTime) {
                    followingEvents.add(new Event(userId, itemId, newEventTime));
                }
            }
        }
//        int predictionLen = 0;
        while (!followingEvents.isEmpty() && generatedEvents.size() < MAX_PREDICTION_LEN) {
//            ++predictionLen;
            final Event curEvent = followingEvents.poll();
            if (curEvent.getTs() >= spanStartTime) {
                generatedEvents.add(curEvent);
            }
            model.accept(curEvent);

            final double newEventTime = curEvent.getTs() + model.timeDelta(curEvent.userId(), curEvent.itemId());
//            System.out.println(model.timeDelta(curEvent.userId(), curEvent.itemId()));
            if (curEvent.getTs() <= newEventTime && newEventTime <= spanEndTime) {
                followingEvents.add(new Event(curEvent.userId(), curEvent.itemId(), newEventTime));
            } else {
                System.out.println(curEvent.getTs() + " " + spanEndTime + " " + newEventTime);
            }
        }
//        System.out.println(generatedEvents.size());
        return generatedEvents;
    }

    public void writeHistory(Path path, List<Event> history) throws IOException {
        final String historyStr = history.stream()
                .map(event -> event.getTs() + " \t" + event.getPair())
                .collect(Collectors.joining("\n", "", "\n"));
        Files.write(path, historyStr.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    public double spuDiffByPair(TLongDoubleMap firstSPUs, TLongDoubleMap secondSPUs) {
        return Arrays.stream(itemsUsers.keys())
                .mapToDouble(itemId -> Math.abs(firstSPUs.get(itemId) - secondSPUs.get(itemId)))
                .average().orElse(-1);
    }

    public double getMeanSpuTarget() {
        return targetPairwiseSPUMean;
    }

    private static void writePairwiseSpus(Path logPath, TLongDoubleMap spus, long[] keys) throws IOException {
        if (logPath != null) {
            final String spusStr = Arrays.stream(keys)
                    .mapToDouble(spus::get)
                    .mapToObj(String::valueOf)
                    .collect(Collectors.joining("\t"));
            Files.write(logPath, (spusStr + '\n').getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }
    }

    public void writeTargetSpus() throws IOException {
        writePairwiseSpus(spuTrainPath, pairwiseHistorySpu(trainData), relevantPairsArray);
        writePairwiseSpus(spuTestPath, targetPairwiseSPU, relevantPairsArray);
    }

    public void writePairNames(Path path, Map <Integer, String> itemIdToName, Map<Integer, String> userIdToName)
            throws IOException {
        if (path != null) {
            final String itemsStr = Arrays.stream(relevantPairsArray)
                    .mapToInt(Util::extractItemId)
                    .mapToObj(itemIdToName::get)
                    .collect(Collectors.joining("\t", "", "\n"));
            final String usersStr = Arrays.stream(relevantPairsArray)
                    .mapToInt(Util::extractUserId)
                    .mapToObj(userIdToName::get)
                    .collect(Collectors.joining("\t", "", "\n"));
            Files.write(path, (itemsStr + usersStr).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        }
    }

    public void writeSpuPairNames(Map <Integer, String> itemIdToName, Map<Integer, String> userIdToName)
            throws IOException {
        writePairNames(spuTrainPath, itemIdToName, userIdToName);
        writePairNames(spuTestPath, itemIdToName, userIdToName);
    }

    public void writeLambdaPairNames(Map <Integer, String> itemIdToName, Map<Integer, String> userIdToName)
            throws IOException {
        writePairNames(lambdaInitialPath, itemIdToName, userIdToName);
        writePairNames(lambdaTrainedPath, itemIdToName, userIdToName);
    }

    private byte[] lambdasBytes(Model.Applicable applicable) {
        return Arrays.stream(relevantPairsArray)
                .mapToDouble(pair -> applicable.getLambda(Util.extractUserId(pair), Util.extractItemId(pair)))
                .mapToObj(String::valueOf)
                .collect(Collectors.joining("\t", "", "\n"))
                .getBytes();
    }

    public void writeLambdas(Model model) throws IOException {
        final Model.Applicable applicable = model.getApplicable();
        Files.write(lambdaInitialPath, lambdasBytes(applicable), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
        applicable.fit(trainData);
        Files.write(lambdaTrainedPath, lambdasBytes(applicable), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
    }

//    public void writeLambdas(Path outPath, Map <Integer, String> itemIdToName, Map<Integer, String> userIdToName,
//                             Model.Applicable applicable) throws IOException {
//        if (outPath == null) {
//            return;
//        }
//        writePairNames(outPath, itemIdToName, userIdToName);
//        final String lambdasStr = Arrays.stream(relevantPairsArray)
//                .mapToDouble(pair -> applicable.getLambda(Util.extractUserId(pair), Util.extractItemId(pair)))
//                .mapToObj(String::valueOf)
//                .collect(Collectors.joining("\t", "", "\n"));
//        Files.write(outPath, lambdasStr.getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
//    }

    public class Summary {
        private final double testReturnTime;
        private final double trainReturnTime;
        private final double recommendMae;
        private final double spusDiffByItem;
        private final double spusMean;
        private final double spusMeanDiff;
        private final TLongDoubleMap spusTest;
        private final TLongDoubleMap spusTrain;

        public Summary(double testReturnTime, double trainReturnTime, double recommendMae, double spusDiffByItem,
                       double spusMean, double spusMeanDiff, TLongDoubleMap spusTest, TLongDoubleMap spusTrain) {
            this.testReturnTime = testReturnTime;
            this.trainReturnTime = trainReturnTime;
            this.recommendMae = recommendMae;
            this.spusDiffByItem = spusDiffByItem;
            this.spusMean = spusMean;
            this.spusMeanDiff = spusMeanDiff;
            this.spusTest = spusTest;
            this.spusTrain = spusTrain;
        }

        @Override
        public String toString() {
            return String.format("test_return_time = %f, train_return_time = %f, recommendation_mae = %f, " +
                            "SPU error = %f, mean SPU = %f, mean SPU error = %f",
                    testReturnTime, trainReturnTime, recommendMae, spusDiffByItem, spusMean, spusMeanDiff);
        }

        public void writeSpus() throws IOException {
            MetricsCalculator.writePairwiseSpus(spuTestPath, spusTest, relevantPairsArray);
            MetricsCalculator.writePairwiseSpus(spuTrainPath, spusTrain, relevantPairsArray);
        }
    }

    public Summary calculateSummary(Model model) throws ExecutionException, InterruptedException {
        final double[] returnTimeMaes = new double[2];
        final ForkJoinTask returnTimeTask = pool.submit(() -> {
            final Model.Applicable applicable = model.getApplicable();
            returnTimeMaes[0] = returnTimeMae(applicable, trainData);
            returnTimeMaes[1] = returnTimeMae(applicable, testData);
        });
        final ForkJoinTask<Double> recommendMaeTask = pool.submit(
                () -> itemRecommendationMae(model));
        final ForkJoinTask<TLongDoubleMap> spusTestTask = pool.submit(() -> {
            final List<Event> prediction =
                    predictSpan(model.getApplicable(trainData), lastTrainEvents, splitTime, endTime);
            return pairwiseHistorySpu(prediction);
        });
        final ForkJoinTask<TLongDoubleMap> spusTrainTask = pool.submit(() -> {
            final List<Event> prediction =
                    predictSpan(model.getApplicable(), beginningTimes, startTime, splitTime);
            return pairwiseHistorySpu(prediction);
        });

        returnTimeTask.join();
        final double trainReturnTime = returnTimeMaes[0];
        final double testReturnTime = returnTimeMaes[1];
        final TLongDoubleMap spusTest = spusTestTask.get();
        final TLongDoubleMap spusTrain = spusTrainTask.get();
        final double spusMean = Arrays.stream(spusTest.values()).average().orElse(-1);
        final double spusMeanDiff = Math.abs(targetPairwiseSPUMean - spusMean);
        final double spusDiffByItem = spuDiffByPair(targetPairwiseSPU, spusTest);
        final double recommendMae = recommendMaeTask.get();
        return new Summary(testReturnTime, trainReturnTime, recommendMae, spusDiffByItem, spusMean, spusMeanDiff,
                spusTest, spusTrain);
    }
}
