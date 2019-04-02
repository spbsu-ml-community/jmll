package com.expleague.erc;

import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.*;
import java.lang.Math;
import java.util.stream.Collectors;

public class MetricsCalculator {
    private static final double LIFE_DELAY = 0.;

    private final List<Event> trainData;
    private final List<Event> testData;
    private final int[] itemIds;
    private final double splitTime;
    private final double endTime;
    private final TIntDoubleMap targetSPU;
    private final TIntObjectMap<TIntSet> itemsUsers;
    private final TIntObjectMap<int[]> itemsUsersArrays;
    private final TIntDoubleMap lastTrainEvents;

    public MetricsCalculator(List<Event> trainData, List<Event> testData) {
        this.trainData = trainData;
        this.testData = testData;
        splitTime = trainData.get(trainData.size() - 1).getTs();
        endTime = testData.get(testData.size() - 1).getTs();
        itemsUsers = selectUsers();
        this.itemIds = itemsUsers.keys();
        itemsUsersArrays = toArrays(itemsUsers);
        targetSPU = spusOnHistory(testData);

        lastTrainEvents = new TIntDoubleHashMap();
        for (Event event: trainData) {
            lastTrainEvents.put(event.userId(), event.getTs());
        }
    }

    private TIntObjectMap<TIntSet> selectUsers() {
        final Map<Integer, Set<Integer>> itemsTrainUsers = trainData.stream()
                .collect(Collectors.groupingBy(Event::itemId, Collectors.mapping(Event::userId, Collectors.toSet())));
        final Map<Integer, Set<Integer>> itemsTestUsers = testData.stream()
                .collect(Collectors.groupingBy(Event::itemId, Collectors.mapping(Event::userId, Collectors.toSet())));
        TIntObjectMap<TIntSet> itemsUsers = new TIntObjectHashMap<>();
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

    private static TIntObjectMap<int[]> toArrays(TIntObjectMap<TIntSet> itemsUsers) {
        final TIntObjectMap<int[]> itemsUsersArrays = new TIntObjectHashMap<>();
        for (int itemId: itemsUsers.keySet().toArray()) {
            itemsUsersArrays.put(itemId, itemsUsers.get(itemId).toArray());
        }
        return itemsUsersArrays;
    }

//    private TIntDoubleMap calcTargetSPUs() {
//        TIntDoubleMap targetSPU = new TIntDoubleHashMap();
//        TSynchronizedIntDoubleMap syncTargetSPU = new TSynchronizedIntDoubleMap(targetSPU);
//        Map<Integer, List<Event>> testEventsByItem = testData.stream().collect(Collectors.groupingBy(Event::itemId));
//        Arrays.stream(itemIds).parallel()
//                .forEach(itemId -> {
//                    List<Event> events = testEventsByItem.get(itemId).stream()
//                            .filter(session -> itemsUsers.get(itemId).contains(session.userId()))
//                            .collect(Collectors.toList());
//                    syncTargetSPU.put(itemId, spuOnData(events, itemId, endTime - splitTime));
//                });
//        return targetSPU;
//    }

    public double returnTimeMae(Model.Applicable model) {
        double errors = 0.;
        long count = 0;
        for (final Event event : testData) {
            count++;
            final double expectedReturnTime = model.timeDelta(event.userId(), event.itemId());
            errors += Math.abs(event.getPrDelta() - expectedReturnTime);
            model.accept(event);
        }
        return errors / count;
    }

    public double itemRecommendationMae(Model.Applicable model) {
        double errors = 0.;
        long count = 0;
        for (Event event: testData) {
            int userId = event.userId();
            List<Integer> itemsByLambda = new ArrayList<>();
            for (final int itemId : model.getItemEmbeddings().keys()) {
                itemsByLambda.add(itemId);
            }
            itemsByLambda.sort(Comparator.comparingDouble(item -> -model.getLambda(userId, item)));
            errors += itemsByLambda.indexOf(event.itemId());
            count++;
            model.accept(event);
        }
        return errors / count;
    }

    public double constantPredictionTimeMae() {
        final double meanItemDelta = trainData.parallelStream()
                .filter(event -> !event.isFinish() && event.getPrDelta() >= 0).collect(Collectors.averagingDouble(Event::getPrDelta));
        return testData.parallelStream()
                .collect(Collectors.averagingDouble(event -> Math.abs(event.getPrDelta() - meanItemDelta)));
    }

//    public static double spuOnData(final List<Event> events, final int itemId, final double timeSpan) {
//        final double maxTimeDiff = 0.5;
//        TIntDoubleMap lastTime = new TIntDoubleHashMap();
//        int sessions = 0;
//        final TIntSet users = new TIntHashSet();
//        for (final Event event : events) {
//            if (event.itemId() != itemId) {
//                continue;
//            }
//            users.add(event.userId());
//            if (!lastTime.containsKey(event.userId()) || event.getTs() - maxTimeDiff > lastTime.get(event.userId())) {
//                lastTime.put(event.userId(), event.getTs());
//                sessions++;
//            }
//        }
//        return (double) sessions / users.size() / timeSpan;
//    }

    public TIntDoubleMap spusOnHistory(List <Event> history) {
        TIntObjectMap<TIntIntMap> itemUserSessions = new TIntObjectHashMap<>();
        TIntObjectMap<TIntDoubleMap> itemUserLifetimes = new TIntObjectHashMap<>();
        for (int itemId: itemIds) {
            TIntIntHashMap curItemSessionsMap = new TIntIntHashMap();
            TIntDoubleHashMap curItemLifetimesMap = new TIntDoubleHashMap();
            for (int userId: itemsUsersArrays.get(itemId)) {
                curItemSessionsMap.put(userId, 0);
                curItemLifetimesMap.put(userId, 0.0);
            }
            itemUserSessions.put(itemId, curItemSessionsMap);
            itemUserLifetimes.put(itemId, curItemLifetimesMap);
        }
        for (Event event: history) {
            final int itemId = event.itemId();
            final int userId = event.userId();
            if (!itemsUsers.containsKey(itemId) || !itemsUsers.get(itemId).contains(userId)) {
                continue;
            }
            TIntIntMap curItemSessions = itemUserSessions.get(itemId);
            curItemSessions.put(userId, curItemSessions.get(userId) + 1);
            TIntDoubleMap curItemLifetimes = itemUserLifetimes.get(itemId);
            curItemLifetimes.put(userId, event.getTs() - splitTime + LIFE_DELAY);
        }

        TIntDoubleMap spus = new TIntDoubleHashMap();
        for (int itemId: itemIds) {
            TIntIntMap curItemSessions = itemUserSessions.get(itemId);
            TIntDoubleMap curItemLifetimes = itemUserLifetimes.get(itemId);
            double curItemSpu = 0.;
            for (int userId: itemsUsersArrays.get(itemId)) {
                curItemSpu += curItemSessions.get(userId) / curItemLifetimes.get(userId);
            }
            curItemSpu /= itemsUsers.get(itemId).size();
            spus.put(itemId, curItemSpu);
        }
        return spus;
    }

//    public double spuOnModel(final int itemId, final Model.Applicable model) {
//        final List<Event> generatedEvents = new ArrayList<>();
//        final Queue<Event> followingEvents = new ArrayDeque<>();
//
//        for (final int userId : itemsUsersArrays.get(itemId)) {
//            double newEventTime = lastTrainEvents.get(userId) + model.timeDelta(userId, itemId);
//            final Event event = new Event(userId, itemId, newEventTime);
//            if (newEventTime <= endTime) {
//                followingEvents.add(event);
//            }
//        }
//        while (!followingEvents.isEmpty()) {
//            final Event curEvent = followingEvents.poll();
//            if (curEvent.getTs() >= splitTime) {
//                generatedEvents.add(curEvent);
//            }
//            model.accept(curEvent);
//            final int userId = curEvent.userId();
//
//            double newEventTime = curEvent.getTs() + model.timeDelta(userId, itemId);
//            final Event nextEvent = new Event(userId, itemId, newEventTime);
//            if (newEventTime <= endTime) {
//                followingEvents.add(nextEvent);
//            }
//        }
//        return spuOnData(generatedEvents, itemId, endTime - splitTime);
//    }

    private List<Event> predictTest(Model.Applicable model) {
        final List<Event> generatedEvents = new ArrayList<>();
        final Queue<Event> followingEvents = new ArrayDeque<>();
        for (int itemId: itemIds) {
            for (int userId: itemsUsersArrays.get(itemId)) {
                double newEventTime = lastTrainEvents.get(userId) + model.timeDelta(userId, itemId);
                final Event event = new Event(userId, itemId, newEventTime);
                if (newEventTime <= endTime) {
                    followingEvents.add(event);
                }
            }
        }
        while (!followingEvents.isEmpty()) {
            final Event curEvent = followingEvents.poll();
            if (curEvent.getTs() >= splitTime) {
                generatedEvents.add(curEvent);
            }
            model.accept(curEvent);

            double newEventTime = curEvent.getTs() + model.timeDelta(curEvent.userId(), curEvent.itemId());
            final Event nextEvent = new Event(curEvent.userId(), curEvent.itemId(), newEventTime);
            if (newEventTime <= endTime) {
                followingEvents.add(nextEvent);
            }
        }
        return generatedEvents;
    }

//    public TIntDoubleMap spusModel(final Model model) {
//        final TIntDoubleHashMap SPUs = new TIntDoubleHashMap();
//        final TSynchronizedIntDoubleMap SPUsSync = new TSynchronizedIntDoubleMap(SPUs);
//        Arrays.stream(itemIds).parallel()
//                .forEach(itemId -> SPUsSync.put(itemId, spuOnModel(itemId, model.getApplicable(trainData))));
//        return SPUs;
//    }

    public double compareSPUs(TIntDoubleMap firstSPUs, TIntDoubleMap secondSPUs) {
        return Arrays.stream(itemsUsers.keys())
                .mapToDouble(itemId -> Math.abs(firstSPUs.get(itemId) - secondSPUs.get(itemId)))
                .average().getAsDouble();
    }

    public void printMetrics(Model model) {
        final double testReturnTime = returnTimeMae(model.getApplicable(trainData));
        final double trainReturnTime = returnTimeMae(model.getApplicable());
        final double recommendMae = itemRecommendationMae(model.getApplicable(trainData));
        final List<Event> prediction = predictTest(model.getApplicable(trainData));
        final TIntDoubleMap SPUsFromModel = spusOnHistory(prediction);
        final double SPUsDiff = compareSPUs(targetSPU, SPUsFromModel);

        System.out.printf("test_return_time = %f, train_return_time = %f, recommendation_mae = %f, SPU error = %f",
                testReturnTime, trainReturnTime, recommendMae, SPUsDiff);
    }
}
