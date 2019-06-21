package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.EventSeq;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;


public class NotLookAheadLambdaStrategy implements LambdaStrategy {
    private final TIntDoubleMap prevUserActionTime;
    private final TIntObjectMap<UserLambda> userLambdas;
    private final TIntObjectMap<TIntDoubleMap> savedLambdas;
    private final TIntObjectMap<TIntObjectMap<Vec>> savedLambdasUserDerivative;
    private final TIntObjectMap<TIntObjectMap<TIntObjectMap<Vec>>> savedLambdasItemDerivative;

    public NotLookAheadLambdaStrategy(final TIntObjectMap<Vec> userEmbeddings, final TIntObjectMap<Vec> itemEmbeddings,
                                      final double beta, final double otherProjectImportance) {
        prevUserActionTime = new TIntDoubleHashMap();
        userLambdas = new TIntObjectHashMap<>();
        savedLambdas = new TIntObjectHashMap<>();
        savedLambdasUserDerivative = new TIntObjectHashMap<>();
        savedLambdasItemDerivative = new TIntObjectHashMap<>();
        for (final int userId : userEmbeddings.keys()) {
            userLambdas.put(userId, new UserLambdaItemSpecific(userEmbeddings.get(userId), itemEmbeddings, beta, otherProjectImportance));
            savedLambdas.put(userId, new TIntDoubleHashMap());
            savedLambdasUserDerivative.put(userId, new TIntObjectHashMap<>());
            savedLambdasItemDerivative.put(userId, new TIntObjectHashMap<>());
        }
    }

    @Override
    public double getLambda(final int userId, final int itemId) {
        if (savedLambdas.get(userId).containsKey(itemId)) {
            return savedLambdas.get(userId).get(itemId);
        }
        double lambda = userLambdas.get(userId).getLambda(itemId);
        savedLambdas.get(userId).put(itemId, lambda);
        return lambda;
    }

    @Override
    public double getLambda(int userId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Vec getLambdaUserDerivative(int userId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Vec getLambdaUserDerivative(final int userId, final int itemId) {
        if (savedLambdasUserDerivative.get(userId).containsKey(itemId)) {
            return savedLambdasUserDerivative.get(userId).get(itemId);
        }
        Vec derivative = userLambdas.get(userId).getLambdaUserDerivative(itemId);
        savedLambdasUserDerivative.get(userId).put(itemId, derivative);
        return derivative;
    }

    @Override
    public TIntObjectMap<Vec> getLambdaItemDerivative(int userId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public TIntObjectMap<Vec> getLambdaItemDerivative(final int userId, final int itemId) {
        if (savedLambdasItemDerivative.get(userId).containsKey(itemId)) {
            return savedLambdasItemDerivative.get(userId).get(itemId);
        }
        TIntObjectMap<Vec> derivatives = userLambdas.get(userId).getLambdaItemsDerivative(itemId);
        savedLambdasItemDerivative.get(userId).put(itemId, derivatives);
        return derivatives;
    }

    @Override
    public void accept(final EventSeq eventSeq) {
        int userId = eventSeq.userId();
        int itemId = eventSeq.itemId();
        double timeDelta = 0.;
        if (prevUserActionTime.containsKey(userId)) {
            timeDelta = eventSeq.getStartTs() - prevUserActionTime.get(userId);
        }
        userLambdas.get(userId).update(itemId, timeDelta);
        savedLambdas.get(userId).put(itemId, userLambdas.get(userId).getLambda(itemId));
        savedLambdasUserDerivative.get(userId).put(itemId, userLambdas.get(userId).getLambdaUserDerivative(itemId));
        savedLambdasItemDerivative.get(userId).put(itemId, userLambdas.get(userId).getLambdaItemsDerivative(itemId));
        prevUserActionTime.put(userId, eventSeq.getStartTs());
    }

    public static class NotLookAheadLambdaStrategyFactory implements LambdaStrategyFactory {
        @Override
        public LambdaStrategy get(final TIntObjectMap<Vec> userEmbeddings, final TIntObjectMap<Vec> itemEmbeddings,
                                  final TIntDoubleMap initialLambdas, double beta, double otherProjectImportance) {
            return new NotLookAheadLambdaStrategy(userEmbeddings, itemEmbeddings, beta, otherProjectImportance);
        }
    }
}
