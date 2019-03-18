package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.Event;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class NotLookAheadLambdaStrategy implements LambdaStrategy {
    private final Map<String, Double> prevUserActionTime;
    private final Map<String, UserLambda> userLambdas;
    private final Map<String, Map<String, Double>> savedLambdas;
    private final Map<String, Map<String, Vec>> savedLambdasUserDerivative;
    private final Map<String, Map<String, Map<String, Vec>>> savedLambdasItemDerivative;

    public NotLookAheadLambdaStrategy(final Map<String, Vec> userEmbeddings, final Map<String, Vec> itemEmbeddings,
                                      final double beta, final double otherProjectImportance) {
        prevUserActionTime = new HashMap<>();
        userLambdas = userEmbeddings.keySet().stream().collect(Collectors.toMap(Function.identity(),
                userId -> new UserLambda(userEmbeddings.get(userId), itemEmbeddings, beta, otherProjectImportance)));
        savedLambdas = userEmbeddings.keySet().stream()
                .collect(Collectors.toMap(Function.identity(), userId -> new HashMap<>()));
        savedLambdasUserDerivative = userEmbeddings.keySet().stream()
                .collect(Collectors.toMap(Function.identity(), userId -> new HashMap<>()));
        savedLambdasItemDerivative = userEmbeddings.keySet().stream()
                .collect(Collectors.toMap(Function.identity(), userId -> new HashMap<>()));
    }

    @Override
    public double getLambda(final String userId, final String itemId) {
        if (savedLambdas.get(userId).containsKey(itemId)) {
            return savedLambdas.get(userId).get(itemId);
        }
        double lambda = userLambdas.get(userId).getLambda(itemId);
        savedLambdas.get(userId).put(itemId, lambda);
        return lambda;
    }

    @Override
    public Vec getLambdaUserDerivative(final String userId, final String itemId) {
        if (savedLambdasUserDerivative.get(userId).containsKey(itemId)) {
            return savedLambdasUserDerivative.get(userId).get(itemId);
        }
        Vec derivative = userLambdas.get(userId).getLambdaUserDerivative(itemId);
        savedLambdasUserDerivative.get(userId).put(itemId, derivative);
        return derivative;
    }

    @Override
    public Map<String, Vec> getLambdaItemDerivative(final String userId, final String itemId) {
        if (savedLambdasItemDerivative.get(userId).containsKey(itemId)) {
            return savedLambdasItemDerivative.get(userId).get(itemId);
        }
        Map<String, Vec> derivatives = userLambdas.get(userId).getLambdaItemsDerivative(itemId);
        savedLambdasItemDerivative.get(userId).put(itemId, derivatives);
        return derivatives;
    }

    @Override
    public void accept(final Event event) {
        String userId = event.userId();
        String itemId = event.itemId();
        double timeDelta = 0.;
        if (prevUserActionTime.containsKey(userId)) {
            timeDelta = event.getTs() - prevUserActionTime.get(userId);
        }
        userLambdas.get(userId).update(itemId, timeDelta);
        savedLambdas.get(userId).put(itemId, userLambdas.get(userId).getLambda(itemId));
        savedLambdasUserDerivative.get(userId).put(itemId, userLambdas.get(userId).getLambdaUserDerivative(itemId));
        savedLambdasItemDerivative.get(userId).put(itemId, userLambdas.get(userId).getLambdaItemsDerivative(itemId));
        prevUserActionTime.put(userId, event.getTs());
    }

    public static class NotLookAheadLambdaStrategyFactory implements LambdaStrategyFactory {
        @Override
        public LambdaStrategy get(Map<String, Vec> userEmbeddings, Map<String, Vec> itemEmbeddings, double beta, double otherProjectImportance) {
            return new NotLookAheadLambdaStrategy(userEmbeddings, itemEmbeddings, beta, otherProjectImportance);
        }
    }
}
