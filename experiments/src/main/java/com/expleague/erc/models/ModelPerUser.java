package com.expleague.erc.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TLongDoubleHashMap;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

import static java.lang.Math.exp;
import static java.lang.Math.max;

public class ModelPerUser extends Model {
    public ModelPerUser(int dim, double beta, double eps, double otherItemImportance,
                        DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                        LambdaStrategyFactory lambdaStrategyFactory, TIntObjectMap<Vec> usersEmbeddingsPrior,
                        TIntObjectMap<Vec> itemsEmbeddingsPrior) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                usersEmbeddingsPrior, itemsEmbeddingsPrior);
    }

    @Override
    public void logLikelihoodDerivative(final List<Event> events,
                                        final TIntObjectMap<Vec> userDerivatives,
                                        final TIntObjectMap<Vec> itemDerivatives) {
//        final double observationEnd = events.get(events.size() - 1).getTs();
        for (final int userId : userIds.toArray()) {
            userDerivatives.put(userId, new ArrayVec(dim));
        }
        for (final int itemId : itemIds.toArray()) {
            itemDerivatives.put(itemId, new ArrayVec(dim));
        }
        final LambdaStrategy lambdaStrategy =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
//        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
//        final TIntDoubleMap userLastVisitTimes = new TIntDoubleHashMap();
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final double delta = session.getDelta();
            if (0 < delta && delta < DataPreprocessor.CHURN_THRESHOLD) {
                updateDerivativeInnerEvent(lambdaStrategy, session.userId(), delta, userDerivatives,
                        itemDerivatives);
            }
            session.getEventSeqs().forEach(lambdaStrategy::accept);
        }
//        for (long pairId: lastVisitTimes.keys()) {
//            final int userId = Util.extractUserId(pairId);
//            final int itemId = Util.extractItemId(pairId);
//            updateDerivativeLastEvent(lambdaStrategy, userId, itemId, observationEnd - lastVisitTimes.get(pairId),
//                    userDerivatives, itemDerivatives);
//        }
    }

    protected void updateDerivativeInnerEvent(LambdaStrategy lambdasByItem, int userId, double timeDelta,
                                              TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives) {
        final double lam = lambdasByItem.getLambda(userId);
        final Vec lamDU = lambdasByItem.getLambdaUserDerivative(userId);
        final TIntObjectMap<Vec> lamDI = lambdasByItem.getLambdaItemDerivative(userId);
        final double tLam = lambdaTransform.applyAsDouble(lam);
        final double tau = max(timeDelta, eps);

        final double expPlus = exp(-tLam * (tau + eps));
        final double expMinus = exp(-tLam * (tau - eps));
        final double lamDT = lambdaDerivativeTransform.applyAsDouble(lam);
        final double commonPart = lamDT * ((tau + eps) * expPlus - (tau - eps) * expMinus) / (-expPlus + expMinus);

        if (!Double.isNaN(commonPart)) {
            VecTools.scale(lamDU, commonPart);
            VecTools.append(userDerivatives.get(userId), lamDU);
            for (TIntObjectIterator<Vec> it = lamDI.iterator(); it.hasNext(); ) {
                it.advance();
                final Vec derivative = it.value();
                VecTools.scale(derivative, commonPart);
                VecTools.append(itemDerivatives.get(it.key()), derivative);
            }
        }
    }

    private class ApplicableImpl implements ApplicableModel {
        private final LambdaStrategy lambdaStrategy;

        private ApplicableImpl() {
            lambdaStrategy = lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        }

        @Override
        public void accept(final EventSeq eventSeq) {
            lambdaStrategy.accept(eventSeq);
        }

        @Override
        public double getLambda(int userId) {
            return lambdaTransform.applyAsDouble(lambdaStrategy.getLambda(userId));
        }

        @Override
        public double getLambda(final int userId, final int itemId) {
            throw new UnsupportedOperationException();
        }

        @Override
        public double timeDelta(final int userId, final double time) {
            return 1 / getLambda(userId);
        }

        @Override
        public double timeDelta(final int userId, final int itemId) {
            throw new UnsupportedOperationException();
        }

        @Override
        public double probabilityBeforeX(int userId, double x) {
            return 1 - exp(-getLambda(userId) * x);
        }

        @Override
        public double probabilityBeforeX(int userId, int itemId, double x) {
            return 1 - exp(-getLambda(userId, itemId) * x);
        }
    }

    public ApplicableModel getApplicable(final List<Event> events) {
        ApplicableModel applicable = new ApplicableImpl();
        if (events != null) {
            applicable.fit(events);
        }
        return applicable;
    }

    public ApplicableModel getApplicable() {
        return new ApplicableImpl();
    }
}
