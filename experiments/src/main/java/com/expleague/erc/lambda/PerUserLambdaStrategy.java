package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.EventSeq;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;

public class PerUserLambdaStrategy implements LambdaStrategy {
    private final TIntObjectMap<UserLambdaSingle> userLambdas;

    public PerUserLambdaStrategy(final TIntObjectMap<Vec> userEmbeddings, final TIntObjectMap<Vec> itemEmbeddings,
                                 final double beta, final TIntDoubleMap initialValues) {
        userLambdas = new TIntObjectHashMap<>();
        for (final int userId : userEmbeddings.keys()) {
            userLambdas.put(userId, new UserLambdaSingle(userEmbeddings.get(userId), itemEmbeddings, beta,
                    initialValues.get(userId)));
        }
    }

    @Override
    public double getLambda(final int userId) {
        return userLambdas.get(userId).getLambda();
    }

    @Override
    public double getLambda(int userId, int itemId) {
        return getLambda(userId);
    }

    @Override
    public Vec getLambdaUserDerivative(int userId) {
        return userLambdas.get(userId).getLambdaUserDerivative();
    }

    @Override
    public Vec getLambdaUserDerivative(final int userId, final int itemId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public TIntObjectMap<Vec> getLambdaItemDerivative(int userId) {
        return userLambdas.get(userId).getLambdaItemsDerivative();
    }

    @Override
    public TIntObjectMap<Vec> getLambdaItemDerivative(final int userId, final int itemId) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void accept(final EventSeq eventSeq) {
        final UserLambda userLambda = userLambdas.get(eventSeq.userId());
        if (eventSeq.getDelta() > DataPreprocessor.CHURN_THRESHOLD) {
            userLambda.reset();
        }
        userLambda.update(eventSeq.itemId(), eventSeq.getDelta());
    }

    public static class Factory implements LambdaStrategyFactory {
        private final TIntDoubleMap initialValues;

        public Factory(TIntDoubleMap initialValues) {
            this.initialValues = initialValues;
        }

        @Override
        public LambdaStrategy get(final TIntObjectMap<Vec> userEmbeddings, final TIntObjectMap<Vec> itemEmbeddings,
                                  double beta, double otherProjectImportance) {
            return new PerUserLambdaStrategy(userEmbeddings, itemEmbeddings, beta, initialValues);
        }
    }
}
