package com.expleague.erc.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.random.FastRandom;
import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.hash.TIntHashSet;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;
import static com.expleague.erc.GammaUtils.*;

import static java.lang.Math.*;

public class ModelUserK extends Model {
    private static final long MAX_K = 10L;
    private static final long MIN_K = 1L;
    private static final double VARIANCE_SHARE = .1;

    private TIntIntMap userKs;
    private TIntDoubleMap userBaseLambdas;

    public ModelUserK(int dim, double beta, double eps, double otherItemImportance, DoubleUnaryOperator lambdaTransform,
                      DoubleUnaryOperator lambdaDerivativeTransform, LambdaStrategyFactory lambdaStrategyFactory) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
    }

//    public ModelUserK(int dim, double beta, double eps, double otherItemImportance, DoubleUnaryOperator lambdaTransform,
//                      DoubleUnaryOperator lambdaDerivativeTransform, LambdaStrategyFactory lambdaStrategyFactory,
//                      TIntObjectMap<Vec> usersEmbeddingsPrior, TIntObjectMap<Vec> itemsEmbeddingsPrior) {
//        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
//                usersEmbeddingsPrior, itemsEmbeddingsPrior);
//    }

    @Override
    public void initModel(final List<Event> events) {
        makeInitialEmbeddings(events);
        initIds();
        userKs = new TIntIntHashMap();
        userBaseLambdas = new TIntDoubleHashMap();
        calcUserParams(events, userBaseLambdas, userKs);
        isInit = true;
    }

    private static void calcUserParams(List<Event> history, TIntDoubleMap userBaseLambdas, TIntIntMap userKs) {
        final TIntDoubleMap userMeans = new TIntDoubleHashMap();
        userMeans.putAll(history.stream()
                .filter(event -> event.getPrDelta() >= 0)
                .collect(Collectors.groupingBy(Event::userId, Collectors.averagingDouble(Event::getPrDelta))));
        final TIntDoubleMap userVariances = new TIntDoubleHashMap();
        userVariances.putAll(history.stream()
                .filter(event -> event.getPrDelta() >= 0)
                .collect(Collectors.groupingBy(
                        Event::userId,
                        Collectors.averagingDouble(event -> {
                            final double diff = event.getPrDelta() - userMeans.get(event.userId());
                            return diff * diff;
                        })
                ))
        );
        final Map<Integer, Long> eventNumbers = history.stream()
                .filter(event -> event.getPrDelta() >= 0)
                .collect(Collectors.groupingBy(Event::userId, Collectors.counting()));
        userMeans.forEachEntry((userId, userMean) -> {
            if (eventNumbers.get(userId) == 1) {
                return true;
            }
            final double userVariance = userVariances.get(userId);
//            final double baseLambda = userMean / userVariance;
            final int userK = (int)min(max(MIN_K, round(userMean * userMean / userVariance)), MAX_K);
            final double baseLambda = userK / userMean;
            userBaseLambdas.put(userId, baseLambda);
            userKs.put(userId, userK);
            return true;
        });
        final double totalMean = Arrays.stream(userMeans.values()).average().orElse(-1);
        final double totalVariance = Arrays.stream(userVariances.values()).average().orElse(-1);
//        final double defaultLambda = totalMean / totalVariance;
        final int defaultK = (int)min(max(MIN_K, round(totalMean * totalMean / totalVariance)), MAX_K);
        final double defaultLambda = defaultK / totalMean;
        history.stream()
                .filter(event -> event.getPrDelta() < 0)
                .mapToInt(Event::userId)
                .filter(userId -> !userKs.containsKey(userId))
                .forEach(userId -> {
                    userBaseLambdas.put(userId, defaultLambda);
                    userKs.put(userId, defaultK);
                });
    }

    @Override
    protected void makeInitialEmbeddings(final List<Event> history) {
        final double targetVariance = Arrays.stream(userBaseLambdas.values()).min().orElse(-1) * VARIANCE_SHARE;
        final double embeddingMean = sqrt(targetVariance) / dim;
        final FastRandom randomGenerator = new FastRandom();
        userIds = new TIntHashSet();
        itemIds = new TIntHashSet();
        for (final Event event : history) {
            userIds.add(event.userId());
            itemIds.add(event.itemId());
        }
        userIds.forEach(userId -> {
            userEmbeddings.put(userId, getGaussianEmbedding(randomGenerator, embeddingMean, dim));
            return true;
        });
        itemIds.forEach(itemId -> {
            itemEmbeddings.put(itemId, getGaussianEmbedding(randomGenerator, embeddingMean, dim));
            return true;
        });
    }

    @Override
    protected void updateDerivativeInnerEvent(LambdaStrategy lambdasByItem, int userId, int itemId, double timeDelta,
                                              TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives,
                                              TIntDoubleMap initialLambdasDerivatives) {
        final double lam = userBaseLambdas.get(userId) + lambdasByItem.getLambda(userId, itemId);
        final Vec lamDU = lambdasByItem.getLambdaUserDerivative(userId, itemId);
        final TIntObjectMap<Vec> lamDI = lambdasByItem.getLambdaItemDerivative(userId, itemId);
        final double tLam = lambdaTransform.applyAsDouble(lam);
        final double tDLam = lambdaDerivativeTransform.applyAsDouble(lam);
        final double tau = max(timeDelta, eps);

        final double upB = tau + eps;
        final double lowB = tau - eps;
        final double upBLam = upB * tLam;
        final double lowBLam = lowB * tLam;
        final int k = userKs.get(userId);
        final double commonPart = tDLam * (pow(upBLam, k - 1) * exp(-upBLam) * upB - pow(lowBLam, k - 1) * exp(-lowBLam) * lowB) /
                (lowerGamma(k, upBLam) - lowerGamma(k, lowBLam));

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
    protected void updateDerivativeLastEvent(LambdaStrategy lambdasByItem, int userId, int itemId, double tau,
                                             TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives) {}

    private class ApplicableImpl implements ApplicableModel {
        private final LambdaStrategy lambdaStrategy;

        private ApplicableImpl() {
            lambdaStrategy = lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, initialLambdas,
                    beta, otherItemImportance);
        }

        @Override
        public void accept(final EventSeq eventSeq) {
            lambdaStrategy.accept(eventSeq);
        }

        @Override
        public double getLambda(int userId) {
            throw new UnsupportedOperationException();
        }

        @Override
        public double getLambda(int userId, int itemId) {
            return lambdaTransform.applyAsDouble(userBaseLambdas.get(userId) + lambdaStrategy.getLambda(userId, itemId));
        }

        @Override
        public double timeDelta(int userId, int itemId) {
            return userKs.get(userId) / getLambda(userId, itemId);
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
            return regularizedGammaP(userKs.get(userId), x * getLambda(userId, itemId));
        }
    }

    @Override
    public ApplicableModel getApplicable() {
        return new ApplicableImpl();
    }

    @Override
    public ApplicableModel getApplicable(List<Event> events) {
        final ApplicableImpl applicable = new ApplicableImpl();
        applicable.fit(events);
        return applicable;
    }

    public TIntIntMap getUserKs() {
        return userKs;
    }

    public TIntDoubleMap getUserBaseLambdas() {
        return userBaseLambdas;
    }
}
