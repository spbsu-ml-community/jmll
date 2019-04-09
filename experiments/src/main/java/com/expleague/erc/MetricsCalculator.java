package com.expleague.erc;

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

    private final List<Event> trainData;
    private final List<Event> testData;
    private final int[] itemIds;
    private final double splitTime;
    private final double endTime;
    private final TIntDoubleMap targetSPU;
    private final TLongDoubleMap targetPairwiseSPU;
    private final double targetPairwiseSPUMean;
    private final TIntObjectMap<TIntSet> itemsUsers;
    private final TLongSet relevantPairs;
    private final long[] relevantPairsArray;
    private final TIntObjectMap<int[]> itemsUsersArrays;
    private final TIntDoubleMap lastTrainEvents;
    private final ForkJoinPool pool;
    private final Path spuLogPath;

    public MetricsCalculator(List<Event> trainData, List<Event> testData, Path spuLogPath) {
        this.trainData = trainData;
        this.testData = testData;
        splitTime = trainData.get(trainData.size() - 1).getTs();
        endTime = testData.get(testData.size() - 1).getTs();
        itemsUsers = selectUsers(trainData, testData);
        relevantPairs = selectPairs(itemsUsers);
        relevantPairsArray = pairsToArray(relevantPairs);
        this.itemIds = itemsUsers.keys();
        itemsUsersArrays = toArrays(itemsUsers);
        targetSPU = spusOnHistory(testData, splitTime);
        targetPairwiseSPU = pairwiseHistorySpu(testData);
        targetPairwiseSPUMean = Arrays.stream(targetPairwiseSPU.values()).average().orElse(-1);
        this.spuLogPath = spuLogPath;

        lastTrainEvents = new TIntDoubleHashMap();
        for (Event event: trainData) {
            lastTrainEvents.put(event.userId(), event.getTs());
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

    public double itemRecommendationMae(Model.Applicable model) {
        long errorsSum = 0;
        long count = 0;
        int[] allItemIds = model.getItemEmbeddings().keys();
        for (Event event: testData) {
            int userId = event.userId();
            final double actualLambda = model.getLambda(userId, event.itemId());
            for (int i: allItemIds) {
                if (model.getLambda(userId, i) > actualLambda) {
                    errorsSum++;
                }
            }
            count++;
            model.accept(event);
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
            pairSpus.put(pair, sessions / (pairDeathTimes.get(pair) - pairBirthTimes.get(pair)));
        }
        return pairSpus;
    }

    public TIntDoubleMap spusOnHistory(List <Event> history, double startTime) {
        final TIntObjectMap<TIntIntMap> itemUserSessions = new TIntObjectHashMap<>();
        final TIntObjectMap<TIntDoubleMap> itemUserLifetimes = new TIntObjectHashMap<>();
        final TIntObjectMap<TIntDoubleMap> itemUserBirthTimes = new TIntObjectHashMap<>();
        for (int itemId: itemIds) {
            final TIntIntHashMap curItemSessionsMap = new TIntIntHashMap();
            final TIntDoubleHashMap curItemLifetimesMap = new TIntDoubleHashMap();
            for (int userId: itemsUsersArrays.get(itemId)) {
                curItemSessionsMap.put(userId, 0);
                curItemLifetimesMap.put(userId, 0.0);
            }
            itemUserSessions.put(itemId, curItemSessionsMap);
            itemUserLifetimes.put(itemId, curItemLifetimesMap);
            itemUserBirthTimes.put(itemId, new TIntDoubleHashMap());
        }
        for (Event event: history) {
            final int itemId = event.itemId();
            final int userId = event.userId();
            if (!itemsUsers.containsKey(itemId) || !itemsUsers.get(itemId).contains(userId)) {
                continue;
            }
            itemUserLifetimes.get(itemId).put(userId, event.getTs() - startTime);
            final TIntDoubleMap curItemBirthTimes = itemUserBirthTimes.get(itemId);
            if (!curItemBirthTimes.containsKey(userId)) {
//                curItemBirthTimes.put(userId, 0);
                curItemBirthTimes.put(userId, event.getTs() - startTime);
            }
        }
//        for (int itemId : itemIds) {
//            final TIntDoubleMap curItemLifetimes = itemUserLifetimes.get(itemId);
//            for (int userId: itemsUsersArrays.get(itemId)) {
//                curItemLifetimes.put(userId, curItemLifetimes.get(userId) / 2);
//            }
//        }
        for (Event event: history) {
            final int itemId = event.itemId();
            final int userId = event.userId();
            if (!itemsUsers.containsKey(itemId) || !itemsUsers.get(itemId).contains(userId)) {
                continue;
            }
//            if (event.getTs() - startTime >=
//                    (itemUserLifetimes.get(itemId).get(userId) + itemUserBirthTimes.get(itemId).get(userId)) / 2) {
                final TIntIntMap curItemSessions = itemUserSessions.get(itemId);
                curItemSessions.put(userId, curItemSessions.get(userId) + 1);
//            }
        }

        final TIntDoubleMap spus = new TIntDoubleHashMap();
        for (int itemId: itemIds) {
            final TIntIntMap curItemSessions = itemUserSessions.get(itemId);
            final TIntDoubleMap curItemLifetimes = itemUserLifetimes.get(itemId);
            double curItemSpu = 0.;
            for (int userId: itemsUsersArrays.get(itemId)) {
                double lifetime = (curItemLifetimes.get(userId) - itemUserBirthTimes.get(itemId).get(userId) + DAY - EPS) / DAY;
                curItemSpu += lifetime != 0. ? curItemSessions.get(userId) / lifetime : 0.;
            }
            curItemSpu /= itemsUsers.get(itemId).size();
            spus.put(itemId, curItemSpu);
        }
        return spus;
    }

    private List<Event> predictTest(Model.Applicable model) {
        final List<Event> generatedEvents = new ArrayList<>();
        final Queue<Event> followingEvents = new ArrayDeque<>();
        for (int itemId: itemIds) {
            for (int userId: itemsUsersArrays.get(itemId)) {
                final double newEventTime = lastTrainEvents.get(userId) + model.timeDelta(userId, itemId);
                if (newEventTime <= endTime) {
                    final Event newEvent = new Event(userId, itemId, newEventTime);
                    followingEvents.add(newEvent);
                }
            }
        }
        while (!followingEvents.isEmpty()) {
            final Event curEvent = followingEvents.poll();
            if (curEvent.getTs() >= splitTime) {
                generatedEvents.add(curEvent);
            }
            model.accept(curEvent);

            final double newEventTime = curEvent.getTs() + model.timeDelta(curEvent.userId(), curEvent.itemId());
            if (curEvent.getTs() <= newEventTime && newEventTime <= endTime) {
                final Event nextEvent = new Event(curEvent.userId(), curEvent.itemId(), newEventTime);
                followingEvents.add(nextEvent);
            }
        }
        return generatedEvents;
    }

    public double spuDiffByPair(TLongDoubleMap firstSPUs, TLongDoubleMap secondSPUs) {
        return Arrays.stream(itemsUsers.keys())
                .mapToDouble(itemId -> Math.abs(firstSPUs.get(itemId) - secondSPUs.get(itemId)))
                .average().orElse(-1);
    }

    public double getMeanSpuTarget() {
        return targetPairwiseSPUMean;
    }

//    private static void writeSpus(Path logPath, TIntDoubleMap spus) throws IOException {
//        if (logPath != null) {
//            final String spusStr = Arrays.stream(spus.keys())
//                    .sorted()
//                    .mapToDouble(spus::get)
//                    .mapToObj(String::valueOf)
//                    .collect(Collectors.joining("\t"));
//            Files.write(logPath, (spusStr + '\n').getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
//        }
//    }

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
        writePairwiseSpus(spuLogPath, pairwiseHistorySpu(trainData), relevantPairsArray);
        writePairwiseSpus(spuLogPath, targetPairwiseSPU, relevantPairsArray);
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

    public void writeLambdas(Path outPath, Map <Integer, String> itemIdToName, Map<Integer, String> userIdToName,
                             Model model) throws IOException {
        if (outPath == null) {
            return;
        }
        writePairNames(outPath, itemIdToName, userIdToName);
        Model.Applicable applicable = model.getApplicable(trainData);
        final String lambdasStr = Arrays.stream(relevantPairsArray)
                .mapToDouble(pair -> applicable.getLambda(Util.extractUserId(pair), Util.extractItemId(pair)))
                .mapToObj(String::valueOf)
                .collect(Collectors.joining("\t", "", "\n"));
        Files.write(outPath, lambdasStr.getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
    }

    public class Summary {
        private final double testReturnTime;
        private final double trainReturnTime;
        private final double recommendMae;
        private final double spusDiffByItem;
        private final double spusMean;
        private final double spusMeanDiff;
        private final TLongDoubleMap spus;

        public Summary(double testReturnTime, double trainReturnTime, double recommendMae, double spusDiffByItem,
                       double spusMean, double spusMeanDiff, TLongDoubleMap spus) {
            this.testReturnTime = testReturnTime;
            this.trainReturnTime = trainReturnTime;
            this.recommendMae = recommendMae;
            this.spusDiffByItem = spusDiffByItem;
            this.spusMean = spusMean;
            this.spusMeanDiff = spusMeanDiff;
            this.spus = spus;
        }

        @Override
        public String toString() {
            return String.format("test_return_time = %f, train_return_time = %f, recommendation_mae = %f, " +
                            "SPU error = %f, mean SPU = %f, mean SPU error = %f",
                    testReturnTime, trainReturnTime, recommendMae, spusDiffByItem, spusMean, spusMeanDiff);
        }

        public void writeSpus() throws IOException {
            if (spuLogPath != null) {
                MetricsCalculator.writePairwiseSpus(spuLogPath, spus, relevantPairsArray);
            }
        }
    }

    public Summary calculateSummary(Model model) throws ExecutionException, InterruptedException {
        final double[] returnTimeMaes = new double[2];
        final ForkJoinTask returnTimeTask = pool.submit(() -> {
            Model.Applicable applicable = model.getApplicable();
            returnTimeMaes[0] = returnTimeMae(applicable, trainData);
            returnTimeMaes[1] = returnTimeMae(applicable, testData);
        });
        final ForkJoinTask<Double> recommendMaeTask = pool.submit(
                () -> itemRecommendationMae(model.getApplicable(trainData)));
        final ForkJoinTask<TLongDoubleMap> spusTask = pool.submit(() -> {
            final List<Event> prediction = predictTest(model.getApplicable(trainData));
            return pairwiseHistorySpu(prediction);
        });

        returnTimeTask.join();
        final double trainReturnTime = returnTimeMaes[0];
        final double testReturnTime = returnTimeMaes[1];
        final TLongDoubleMap spus = spusTask.get();
        final double spusMean = Arrays.stream(spus.values()).average().orElse(-1);
        final double spusMeanDiff = Math.abs(targetPairwiseSPUMean - spusMean);
        final double spusDiffByItem = spuDiffByPair(targetPairwiseSPU, spus);
        final double recommendMae = recommendMaeTask.get();
        return new Summary(testReturnTime, trainReturnTime, recommendMae, spusDiffByItem, spusMean, spusMeanDiff, spus);
    }
}
