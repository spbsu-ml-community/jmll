package com.expleague.erc.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

import static java.lang.Math.exp;
import static java.lang.Math.sqrt;

public class ModelDays extends ModelPerUser {
    private static final int DAY_HOURS = 24;

    private final TIntIntMap userDayBorders;
    private final TIntIntMap userDayPeaks;
    private final TIntDoubleMap userDayAvgStarts;
    private final TIntDoubleMap averageOneDayDelta;

    public ModelDays(int dim, double beta, double eps, double otherItemImportance, DoubleUnaryOperator lambdaTransform,
                     DoubleUnaryOperator lambdaDerivativeTransform, LambdaStrategyFactory lambdaStrategyFactory,
                     TIntObjectMap<Vec> usersEmbeddingsPrior, TIntObjectMap<Vec> itemsEmbeddingsPrior,
                     TIntIntMap userDayBorders, TIntIntMap userDayPeaks, TIntDoubleMap userDayAvgStarts,
                     TIntDoubleMap averageOneDayDelta) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                usersEmbeddingsPrior, itemsEmbeddingsPrior);
        this.userDayBorders = userDayBorders;
        this.userDayAvgStarts = userDayAvgStarts;
        this.averageOneDayDelta = averageOneDayDelta;
        this.userDayPeaks = userDayPeaks;
    }

    private double lastDayBorder(double time, int userBorder) {
        double lastBorder = ((int) time / DAY_HOURS) * DAY_HOURS + userBorder;
        if (lastBorder > time) {
            lastBorder -= DAY_HOURS;
        }
        return lastBorder;
    }

    @Override
    public void logLikelihoodDerivative(List<Event> events,
                                        TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives) {
        for (final int userId : userIds.toArray()) {
            userDerivatives.put(userId, new ArrayVec(dim));
        }
        for (final int itemId : itemIds.toArray()) {
            itemDerivatives.put(itemId, new ArrayVec(dim));
        }
        final LambdaStrategy lambdaStrategy =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            if (session.getDelta() < DataPreprocessor.CHURN_THRESHOLD) {
                double pos = lastDayBorder(session.getStartTs(), userDayBorders.get(session.userId()));
                int daysPassed = 0;
                while (pos > session.getStartTs() - session.getDelta()) {
                    daysPassed++;
                    pos -= DAY_HOURS;
                }
                updateDerivativeInnerEvent(lambdaStrategy, session.userId(), daysPassed, userDerivatives, itemDerivatives);
            }
            session.getEventSeqs().forEach(lambdaStrategy::accept);
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
            final double rawPrediction = 1 / getLambda(userId);
            final int daysPrediction = (int) rawPrediction;
            if (time + rawPrediction < lastDayBorder(time, userDayBorders.get(userId)) + DAY_HOURS) {
//            if (daysPrediction == 0) {
                return averageOneDayDelta.get(userId);
            }
            final int predictedDay = (int) (time + daysPrediction * DAY_HOURS) / DAY_HOURS;
//            final double predictedTime = predictedDay * DAY_HOURS + userDayPeaks.get(userId);
            final double predictedTime = predictedDay * DAY_HOURS + userDayAvgStarts.get(userId);
            return predictedTime - time;
        }

        @Override
        public double timeDelta(final int userId, final int itemId) {
            throw new UnsupportedOperationException();
        }

        @Override
        public double probabilityBeforeX(int userId, double x) {
            return 1 - exp(-getLambda(userId) * x / DAY_HOURS);
        }

        @Override
        public double probabilityBeforeX(int userId, int itemId, double x) {
            throw new UnsupportedOperationException();
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

    public static void calcDayPoints(final List<Event> events, final TIntIntMap borders, final TIntIntMap peaks) {
        final TIntObjectMap<long[]> counters = new TIntObjectHashMap<>();
        for (final Event event : events) {
            final int userId = event.userId();
            if (!counters.containsKey(userId)) {
                counters.put(userId, new long[DAY_HOURS]);
            }
            counters.get(userId)[(int) event.getTs() % DAY_HOURS]++;
        }
        counters.forEachEntry((userId, userCounters) -> {
            int argMax = -1;
            int argMin = -1;
            long minCounter = Long.MAX_VALUE;
            long maxCounter = Long.MIN_VALUE;
            for (int i = 0; i < DAY_HOURS; i++) {
                if (userCounters[i] < minCounter) {
                    minCounter = userCounters[i];
                    argMin = i;
                }
                if (userCounters[i] < maxCounter) {
                    maxCounter = userCounters[i];
                    argMax = i;
                }
            }
            borders.put(userId, argMin);
            peaks.put(userId, argMax);
            return true;
        });
    }

    public static TIntDoubleMap calcAvgStarts(final List<Event> events) {
        final TIntIntMap lastDays = new TIntIntHashMap();
        TIntDoubleMap starts = new TIntDoubleHashMap();
        TIntIntMap count = new TIntIntHashMap();
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int userId = session.userId();
            final int curDay = ((int) session.getStartTs() / DAY_HOURS);
            final double curTime = session.getStartTs() - curDay * DAY_HOURS;
            if (!lastDays.containsKey(userId) || lastDays.get(userId) != curDay) {
                starts.adjustOrPutValue(userId, curTime, curTime);
                count.adjustOrPutValue(userId, 1, 1);
                lastDays.put(userId, curDay);
            }
        }
        TIntDoubleMap avgStarts = new TIntDoubleHashMap();
        for (final int userId : starts.keys()) {
            avgStarts.put(userId, starts.get(userId) / count.get(userId));
        }
        return avgStarts;
    }

    public static TIntDoubleMap calcAverageOneDayDelta(final List<Event> events) {
        final TIntIntMap lastDays = new TIntIntHashMap();
        final TIntDoubleMap lastDayTimes = new TIntDoubleHashMap();
        final TIntDoubleMap userDeltas = new TIntDoubleHashMap();
        final TIntIntMap userCounts = new TIntIntHashMap();
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int userId = session.userId();
            final int curDay = ((int) session.getStartTs() / DAY_HOURS);
            final double curTime = session.getStartTs() - curDay * DAY_HOURS;
            if (!lastDays.containsKey(userId)) {
                lastDays.put(userId, curDay);
                lastDayTimes.put(userId, curTime);
                continue;
            }
            final int lastDay = lastDays.get(userId);
            if (lastDay == curDay) {
                final double delta = curTime - lastDayTimes.get(userId);
                userDeltas.adjustOrPutValue(userId, delta, delta);
                userCounts.adjustOrPutValue(userId, 1, 1);
                lastDayTimes.put(userId, curTime);
            } else {
                lastDays.put(userId, curDay);
                lastDayTimes.put(userId, curTime);
            }
        }
        final TIntDoubleMap result = new TIntDoubleHashMap();
        for (final int userId : userDeltas.keys()) {
            result.put(userId, userDeltas.get(userId) / userCounts.get(userId));
        }
        return result;
    }

    public static void makeInitialEmbeddings(int dim, List<Event> history,
                                             TIntObjectMap<Vec> userEmbeddings, TIntObjectMap<Vec> itemEmbeddings) {
        final TIntSet userIds = new TIntHashSet();
        final TIntSet itemIds = new TIntHashSet();
        for (Event event : history) {
            userIds.add(event.userId());
            itemIds.add(event.itemId());
        }

        final double itemDeltaMean = DataPreprocessor.groupToEventSeqs(history).stream()
                .mapToDouble(EventSeq::getDelta)
                .filter(delta -> delta >= 0)
                .map(x -> x / DAY_HOURS)
                .average().orElse(-1);
        final double embeddingMean = sqrt(1 / itemDeltaMean) / dim;
        final FastRandom randomGenerator = new FastRandom();
        userIds.forEach(userId -> {
            userEmbeddings.put(userId, makeEmbedding(randomGenerator, embeddingMean, dim));
            return true;
        });
        itemIds.forEach(itemId -> {
            itemEmbeddings.put(itemId, makeEmbedding(randomGenerator, embeddingMean, dim));
            return true;
        });
    }
}
