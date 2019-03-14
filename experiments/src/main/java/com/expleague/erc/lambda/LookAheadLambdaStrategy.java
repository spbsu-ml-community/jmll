package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class LookAheadLambdaStrategy implements LambdaStrategy {
    private final Map<String, ArrayVec> userEmbeddings;
    private final Map<String, ArrayVec> itemEmbeddings;
    private final double beta;
    private final double otherProjectImportance;
    private final Map<String, Double> prevUserActionTime;
    private final Map<String, UserLambda> userLambdas;

    public LookAheadLambdaStrategy(Map<String, ArrayVec> userEmbeddings, Map<String, ArrayVec> itemEmbeddings,
                                   double beta, double otherProjectImportance) {
        this.userEmbeddings = userEmbeddings;
        this.itemEmbeddings = itemEmbeddings;
        this.beta = beta;
        this.otherProjectImportance = otherProjectImportance;
        prevUserActionTime = new HashMap<>();
        userLambdas = userEmbeddings.keySet().stream().collect(Collectors.toMap(Function.identity(),
                userId -> new UserLambda(userEmbeddings.get(userId), itemEmbeddings, beta, otherProjectImportance)));
    }

    @Override
    public double getLambda(String userId, String itemId) {
        return userLambdas.get(userId).getLambda(itemId);
    }

    @Override
    public ArrayVec getLambdaUserDerivative(String userId, String itemId) {
        return userLambdas.get(userId).getLambdaUserDerivative(itemId);
    }

    @Override
    public Map<String, ArrayVec> getLambdaProjectDerivative(String userId, String itemId) {
        return userLambdas.get(userId).getLambdaProjectDerivative(itemId);
    }

    @Override
    public void accept(Event event) {
        double timeDelta = 0;
        if (prevUserActionTime.containsKey(event.getUid())) {
            timeDelta = event.getTs() - prevUserActionTime.get(event.getUid());
        }
        userLambdas.get(event.getUid()).update(event.getPid(), timeDelta);
        prevUserActionTime.put(event.getUid(), event.getTs());
    }
}
