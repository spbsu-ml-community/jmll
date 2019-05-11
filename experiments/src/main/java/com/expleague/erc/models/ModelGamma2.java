package com.expleague.erc.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TLongDoubleHashMap;
import gnu.trove.set.TLongSet;
import gnu.trove.set.hash.TLongHashSet;

import java.util.List;
import java.util.function.DoubleUnaryOperator;
import static java.lang.Math.*;

public class ModelGamma2 extends Model {

    public ModelGamma2(int dim, double beta, double eps, double otherItemImportance,
                       DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                       LambdaStrategyFactory lambdaStrategyFactory) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
    }

//    public ModelGamma2(int dim, double beta, double eps, double otherItemImportance,
//                       DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
//                       LambdaStrategyFactory lambdaStrategyFactory,
//                       TIntObjectMap<Vec> usersEmbeddingsPrior, TIntObjectMap<Vec> itemsEmbeddingsPrior) {
//        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
//                usersEmbeddingsPrior, itemsEmbeddingsPrior);
//    }

    public double logLikelihood(List<Event> events) {
        final double observationEnd = events.get(events.size() - 1).getTs();
        double logLikelihood = 0.;
        final TLongSet seenPairs = new TLongHashSet();
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
        for (final EventSeq eventSeq : DataPreprocessor.groupToEventSeqs(events)) {
            final int userId = eventSeq.userId();
            final int itemId = eventSeq.itemId();
            final long pairId = eventSeq.getPair();
            if (!seenPairs.contains(pairId)) {
                seenPairs.add(pairId);
                lambdasByItem.accept(eventSeq);
            } else {
                final double lam = lambdasByItem.getLambda(userId, itemId);
                final double tLam = lambdaTransform.applyAsDouble(lam);
                final double prDelta = max(eventSeq.getDelta(), eps);
                final double upBLam = (prDelta + eps) * tLam;
                final double lowBLam = (prDelta - eps) * tLam;
                final double probLog = log(-exp(-upBLam) * (upBLam + 1) + exp(-lowBLam) * (lowBLam + 1));
                if (Double.isFinite(probLog)) {
                    logLikelihood += probLog;
                }
                lambdasByItem.accept(eventSeq);
                lastVisitTimes.put(pairId, eventSeq.getStartTs());
            }
        }
        for (long pairId : lastVisitTimes.keys()) {
            final int userId = Util.extractUserId(pairId);
            final int itemId = Util.extractItemId(pairId);
            final double lam = lambdasByItem.getLambda(userId, itemId);
            final double tLam = lambdaTransform.applyAsDouble(lam);
            final double tau = observationEnd - lastVisitTimes.get(pairId);
            if (tau > 0) {
                final double probabilityLog = log(exp(-tLam * tau) * (tLam * tau + 1));
                if (Double.isFinite(probabilityLog)) {
                    logLikelihood += probabilityLog;
                }
            }
        }
        return logLikelihood;
    }

    @Override
    protected void updateDerivativeInnerEvent(LambdaStrategy lambdasByItem, int userId, int itemId, double timeDelta,
                                              TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives,
                                              TIntDoubleMap initialLambdasDerivatives) {
        final double lam = lambdasByItem.getLambda(userId, itemId);
        final Vec lamDU = lambdasByItem.getLambdaUserDerivative(userId, itemId);
        final TIntObjectMap<Vec> lamDI = lambdasByItem.getLambdaItemDerivative(userId, itemId);
        final double tLam = lambdaTransform.applyAsDouble(lam);
        final double tDLam = lambdaDerivativeTransform.applyAsDouble(lam);
        final double tau = max(timeDelta, eps);

        final double upB = tau + eps;
        final double lowB = tau - eps;
        final double upBLam = upB * tLam;
        final double lowBLam = lowB * tLam;
        final double expUp = exp(-upBLam);
        final double expLow = exp(-lowBLam);
        final double commonPart = tDLam * (
                - upB * expUp * (upBLam + 1) - expUp * upB
                + lowB * expLow * (lowBLam + 1) + expLow * lowB
        ) / (-expUp * (upBLam + 1) + expLow * (lowBLam + 1));

        if (!Double.isNaN(commonPart)) {
            VecTools.scale(lamDU, commonPart);
            VecTools.append(userDerivatives.get(userId), lamDU);
            for (TIntObjectIterator<Vec> it = lamDI.iterator(); it.hasNext();) {
                it.advance();
                final Vec derivative = it.value();
                VecTools.scale(derivative, commonPart);
                VecTools.append(itemDerivatives.get(it.key()), derivative);
            }
        }
    }

    @Override
    protected void updateDerivativeLastEvent(LambdaStrategy lambdasByItem, int userId, int itemId, double tau, TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives) {
//        if (tau > 0) {
//            final Vec lamDU = lambdasByItem.getLambdaUserDerivative(userId, itemId);
//            final TIntObjectMap<Vec> lamDI = lambdasByItem.getLambdaItemDerivative(userId, itemId);
//            final double lam = lambdasByItem.getLambda(userId, itemId);
//            final double tLam = lambdaTransform.applyAsDouble(lam);
//            final double tDLam = lambdaDerivativeTransform.applyAsDouble(lam);
//
//            final double lamTau = tLam * tau;
//            final double expLT = exp(-lamTau);
//            final double lamTau1 = lamTau + 1;
//            final double commonPart = tDLam * (tau * expLT * lamTau1 + expLT * tau) / (-expLT * lamTau1);
//
//            VecTools.scale(lamDU, commonPart);
//            VecTools.append(userDerivatives.get(userId), lamDU);
//            for (TIntObjectIterator<Vec> it = lamDI.iterator(); it.hasNext(); ) {
//                it.advance();
//                final Vec derivative = it.value();
//                VecTools.scale(derivative, commonPart);
//                VecTools.append(itemDerivatives.get(it.key()), derivative);
//            }
//        }
    }

    private class ApplicableImpl implements ApplicableModel {
        private final LambdaStrategy lambdaStrategy;

        private ApplicableImpl() {
            lambdaStrategy = lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        }

        @Override
        public void accept(EventSeq eventSeq) {
            lambdaStrategy.accept(eventSeq);
        }

        @Override
        public double getLambda(int userId) {
            throw new UnsupportedOperationException();
        }

        @Override
        public double getLambda(int userId, int itemId) {
            return lambdaTransform.applyAsDouble(lambdaStrategy.getLambda(userId, itemId));
        }

        @Override
        public double timeDelta(int userId, int itemId) {
            return 2 / getLambda(userId, itemId);
        }

        @Override
        public double timeDelta(final int userId, final double time) {
            throw new UnsupportedOperationException();
        }

        @Override
        public double probabilityBeforeX(int userId, double x) {
            throw new UnsupportedOperationException();
        }

        @Override
        public double probabilityBeforeX(int userId, int itemId, double x) {
            final double tLam = getLambda(userId, itemId);
            return -exp(-tLam * x) * (tLam * x + 1) + 1;
        }

//        @Override
//        public double probabilityInterval(int userId, int itemId, double start, double end) {
//            final double tLam = getLambda(userId, itemId);
//            final double upBLam = end * tLam;
//            final double lowBLam = start * tLam;
//            return -exp(-upBLam) * (upBLam + 1) + exp(-lowBLam) * (lowBLam + 1);
//        }
    }

    @Override
    public ApplicableModel getApplicable() {
        return new ApplicableImpl();
    }

    @Override
    public ApplicableModel getApplicable(List<Event> events) {
        return new ApplicableImpl().fit(events);
    }
}
