package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.EventSeq;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

public class LookAheadLambdaStrategy implements LambdaStrategy {
    private final TIntDoubleMap prevUserActionTime;
    private final TIntObjectMap<UserLambda> userLambdas;

    public LookAheadLambdaStrategy(final TIntObjectMap<Vec> userEmbeddings, final TIntObjectMap<Vec> itemEmbeddings,
                                   final double beta, final double otherProjectImportance) {
        prevUserActionTime = new TIntDoubleHashMap();
        userLambdas = new TIntObjectHashMap<>();
        for (final int userId : userEmbeddings.keys()) {
            userLambdas.put(userId, new UserLambdaItemSpecific(userEmbeddings.get(userId), itemEmbeddings, beta, otherProjectImportance));
        }
    }

    @Override
    public double getLambda(final int userId, final int itemId) {
        return userLambdas.get(userId).getLambda(itemId);
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
        return userLambdas.get(userId).getLambdaUserDerivative(itemId);
    }

    @Override
    public TIntObjectMap<Vec> getLambdaItemDerivative(int userId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public TIntObjectMap<Vec> getLambdaItemDerivative(final int userId, final int itemId) {
        return userLambdas.get(userId).getLambdaItemsDerivative(itemId);
    }

    @Override
    public void accept(final EventSeq eventSeq) {
        double timeDelta = 0;
        if (prevUserActionTime.containsKey(eventSeq.userId())) {
            timeDelta = eventSeq.getStartTs() - prevUserActionTime.get(eventSeq.userId());
        }
        userLambdas.get(eventSeq.userId()).update(eventSeq.itemId(), timeDelta);
        prevUserActionTime.put(eventSeq.userId(), eventSeq.getStartTs());
    }
}
