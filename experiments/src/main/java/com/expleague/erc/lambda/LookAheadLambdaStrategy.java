package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.Event;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class LookAheadLambdaStrategy implements LambdaStrategy {
    private final Map<String, Double> prevUserActionTime;
    private final Map<String, UserLambda> userLambdas;

    public LookAheadLambdaStrategy(final Map<String, Vec> userEmbeddings, final Map<String, Vec> itemEmbeddings,
                                   final double beta, final double otherProjectImportance) {
        prevUserActionTime = new HashMap<>();
        userLambdas = userEmbeddings.keySet().stream().collect(Collectors.toMap(Function.identity(),
                userId -> new UserLambda(userEmbeddings.get(userId), itemEmbeddings, beta, otherProjectImportance)));
    }

    @Override
    public double getLambda(final String userId, final String itemId) {
        return userLambdas.get(userId).getLambda(itemId);
    }

    @Override
    public Vec getLambdaUserDerivative(final String userId, final String itemId) {
        return userLambdas.get(userId).getLambdaUserDerivative(itemId);
    }

    @Override
    public Map<String, Vec> getLambdaItemDerivative(final String userId, final String itemId) {
        return userLambdas.get(userId).getLambdaItemsDerivative(itemId);
    }

    @Override
    public void accept(final Event event) {
        double timeDelta = 0;
        if (prevUserActionTime.containsKey(event.userId())) {
            timeDelta = event.getTs() - prevUserActionTime.get(event.userId());
        }
        userLambdas.get(event.userId()).update(event.itemId(), timeDelta);
        prevUserActionTime.put(event.userId(), event.getTs());
    }
}
