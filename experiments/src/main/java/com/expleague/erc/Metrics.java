package com.expleague.erc;

import java.util.ArrayList;
import java.util.List;
import java.lang.Math;
import java.util.stream.Collectors;

public abstract class Metrics {
    public static double returnTimeMae(Model.Applicable model, List<Event> data) {
        double errors = 0.;
        long count = 0;
        for (final Event event : data) {
//            if (event.getPrDelta() != null && !event.getPrDelta().isNaN() && !event.isFinish()) {
            count++;
            final double expectedReturnTime = model.timeDelta(event.userId(), event.itemId());
            errors += Math.abs(event.getPrDelta() - expectedReturnTime);
            model.accept(event);
//            }
        }
        return errors / count;
    }

    public static double itemRecommendationMae(Model.Applicable model, List<Event> data) {
        double errors = 0.;
        long count = 0;
        for (Event event: data) {
//            if (!event.isFinish()) {
            String userId = event.userId();
            List<String> projectsByLambda = new ArrayList<>(model.getUserEmbeddings().keySet());
            projectsByLambda.sort((projectA, projectB) -> {
                if (model.getLambda(userId, projectA) == model.getLambda(userId, projectB)) {
                    return 0;
                }
                return model.getLambda(userId, projectA) > model.getLambda(userId, projectB) ? -1 : 1;
            });
            errors += projectsByLambda.indexOf(event.itemId());
            count++;
            model.accept(event);
//            }
        }
        return errors / count;
    }

    public static double constantPredictionTimeMae(List<Event> trainData, List<Event> testData) {
        List<Event> relevantEvents = trainData.stream()
                .filter(event -> !event.isFinish() && event.getPrDelta() != null)
                .collect(Collectors.toList());
        double meanItemDelta = relevantEvents.stream().collect(Collectors.averagingDouble(Event::getPrDelta));
        return testData.stream()
                .collect(Collectors.averagingDouble(event -> Math.abs(event.getPrDelta() - meanItemDelta)));
    }

    public static void printMetrics(Model model, List<Event> trainData, List<Event> testData) {
        double testReturnTime = returnTimeMae(model.getApplicable(trainData), testData);
        double trainReturnTime = returnTimeMae(model.getApplicable(), trainData);
        double recommendMae = itemRecommendationMae(model.getApplicable(trainData), testData);
        System.out.printf("test_return_time = %f, train_return_time = %f, recommendation_mae = %f",
                testReturnTime, trainReturnTime, recommendMae);
    }
}
