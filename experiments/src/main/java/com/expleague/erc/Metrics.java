package com.expleague.erc;

import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.lang.Math;
import java.util.stream.Collectors;

public abstract class Metrics {
    public static double returnTimeMae(Model.Applicable model, List<Event> data) {
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

    public static double itemRecommendationMae(Model.Applicable model, List<Event> data) {
        double errors = 0.;
        long count = 0;
        for (Event event: data) {
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

    public static double constantPredictionTimeMae(List<Event> trainData, List<Event> testData) {
        double meanItemDelta = trainData.parallelStream()
                .filter(event -> !event.isFinish() && event.getPrDelta() >= 0).collect(Collectors.averagingDouble(Event::getPrDelta));
        return testData.parallelStream()
                .collect(Collectors.averagingDouble(event -> Math.abs(event.getPrDelta() - meanItemDelta)));
    }

    public static void printMetrics(Model model, List<Event> trainData, List<Event> testData) {
        double testReturnTime = returnTimeMae(model.getApplicable(trainData), testData);
        double trainReturnTime = returnTimeMae(model.getApplicable(), trainData);
        double recommendMae = itemRecommendationMae(model.getApplicable(trainData), testData);
        System.out.printf("test_return_time = %f, train_return_time = %f, recommendation_mae = %f",
                testReturnTime, trainReturnTime, recommendMae);
    }

    public double spuOnData(final List<Event> events, final int itemId) {
        final double maxTimeDiff = 30 * 60;
        TIntDoubleMap lastTime = new TIntDoubleHashMap();
        int sessions = 0;
        final TIntSet users = new TIntHashSet();
        for (final Event event : events) {
            if (event.itemId() != itemId) {
                continue;
            }
            users.add(event.userId());
            if (lastTime.containsKey(event.userId())) {
                lastTime.put(event.userId(), event.getTs());
                sessions++;
            }
            if (event.getTs() - maxTimeDiff > lastTime.get(event.userId())) {
                sessions++;
                lastTime.put(event.userId(), event.getTs());
            }
        }
        return (double) sessions / users.size();
    }

    public double spuOnModel(final double maxTime, final TIntSet users, final int itemId, final Model.Applicable model) {
//        Optional<Double> maxTime = events.stream().filter(x -> x.itemId() == itemId).map(Event::getTs).max(Comparator.comparingDouble(x -> x));
        final List<Event> generatedEvents = new ArrayList<>();
        for (final int userId : users.toArray()) {
            while (true) {
                final Event event = new Event(userId, itemId, model.timeDelta(userId, itemId));
                if (event.getTs() > maxTime) {
                    break;
                }
                generatedEvents.add(event);
                model.accept(event);
            }
        }
        return spuOnData(generatedEvents, itemId);
    }
}
