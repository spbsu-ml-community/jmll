package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class NotLookAheadLambdaStrategy implements LambdaStrategy {
private final Map<String, ArrayVec> userEmbeddings;
    private final Map<String, ArrayVec> itemEmbeddings;
    private final double beta;
    private final double otherItemsImportance;
    private final Map<String, Double> prevUserActionTime;
    private final Map<String, UserLambda> userLambdas;
    private final Map<String, Map<String, Double>> savedLambdas;
    private final Map<String, Map<String, ArrayVec>> savedLambdasUserDerivative;
    private final Map<String, Map<String, Map<String, ArrayVec>>> savedLambdasItemDerivative;

    public NotLookAheadLambdaStrategy(Map<String, ArrayVec> userEmbeddings, Map<String, ArrayVec> itemEmbeddings,
                                      double beta, double otherProjectImportance) {
        this.userEmbeddings = userEmbeddings;
        this.itemEmbeddings = itemEmbeddings;
        this.beta = beta;
        this.otherItemsImportance = otherProjectImportance;
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
    public double getLambda(String userId, String itemId) {
        if (savedLambdas.get(userId).containsKey(itemId)) {
            return savedLambdas.get(userId).get(itemId);
        }
        double lambda = userLambdas.get(userId).getLambda(itemId);
        savedLambdas.get(userId).put(itemId, lambda);
        return lambda;
    }

    @Override
    public ArrayVec getLambdaUserDerivative(String userId, String itemId) {
        if (savedLambdasUserDerivative.get(userId).containsKey(itemId)) {
            return savedLambdasUserDerivative.get(userId).get(itemId);
        }
        ArrayVec derivative = userLambdas.get(userId).getLambdaUserDerivative(itemId);
        savedLambdasUserDerivative.get(userId).put(itemId, derivative);
        return derivative;
    }

    @Override
    public Map<String, ArrayVec> getLambdaProjectDerivative(String userId, String itemId) {
        if (savedLambdasItemDerivative.get(userId).containsKey(itemId)) {
            return savedLambdasItemDerivative.get(userId).get(itemId);
        }
        Map<String, ArrayVec> derivatives = userLambdas.get(userId).getLambdaItemsDerivative(itemId);
        savedLambdasItemDerivative.get(userId).put(itemId, derivatives);
        return derivatives;
    }

    @Override
    public void accept(Event event) {
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
    }
}
