package com.expleague.erc.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntLongMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntLongHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

import static java.lang.Math.exp;

public class ModelDays extends ModelPerUser {
    private static final int DAY_HOURS = 24;

    private final TIntIntMap userDayBorders;
    private final TIntDoubleMap averageOneDayDelta;

    public ModelDays(int dim, double beta, double eps, double otherItemImportance, DoubleUnaryOperator lambdaTransform,
                     DoubleUnaryOperator lambdaDerivativeTransform, LambdaStrategyFactory lambdaStrategyFactory,
                     TIntObjectMap<Vec> usersEmbeddingsPrior, TIntObjectMap<Vec> itemsEmbeddingsPrior,
                     TIntIntMap userDayBorders, TIntDoubleMap averageOneDayDelta) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                usersEmbeddingsPrior, itemsEmbeddingsPrior);
        this.userDayBorders = userDayBorders;
        this.averageOneDayDelta = averageOneDayDelta;
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
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            double pos = lastDayBorder(session.getStartTs(), userDayBorders.get(session.userId()));
            int daysPassed = 0;
            while (pos > session.getStartTs() - session.getDelta()) {
                ++daysPassed;
                pos -= 1.;
            }
            updateDerivativeInnerEvent(lambdasByItem, session.userId(), daysPassed, userDerivatives, itemDerivatives);
            session.getEventSeqs().forEach(lambdasByItem::accept);
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
        public double timeDelta(int userId) {
            final int daysPred = (int)(1 / getLambda(userId) + .5);
            final double td;
            if (daysPred == 0) {
                td = averageOneDayDelta.get(userId);
            } else {
                td = daysPred * DAY_HOURS;
            }
            return td;
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

    public static TIntIntMap calcDayBorders(List<Event> events) {
        final TIntObjectMap<long[]> counters = new TIntObjectHashMap<>();
        for (Event event: events) {
            final int userId = event.userId();
            if (!counters.containsKey(userId)) {
                counters.put(userId, new long[DAY_HOURS]);
            }
            counters.get(userId)[(int)event.getTs() % DAY_HOURS]++;
        }
        final TIntIntMap userBorders = new TIntIntHashMap();
        counters.forEachEntry((userId, userCounters) -> {
            int argMin = -1;
            long minCounter = Long.MAX_VALUE;
            for (int i = 0; i < DAY_HOURS; ++i) {
                if (userCounters[i] < minCounter) {
                    minCounter = userCounters[i];
                    argMin = i;
                }
            }
            userBorders.put(userId, argMin);
            return true;
        });
        return userBorders;
    }

    public static TIntDoubleMap calcAverageOneDayDelta(List<Event> events) {
        final TIntIntMap lastDays = new TIntIntHashMap();
        final TIntLongMap userSessions = new TIntLongHashMap();
        final TIntIntMap daysActive = new TIntIntHashMap();
        for (Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int userId = session.userId();
            userSessions.put(userId, userSessions.get(userId) + 1);
            final int curDay = ((int)session.getStartTs() / DAY_HOURS) * DAY_HOURS;
            final int lastDay = lastDays.get(userId);
            if (lastDay != curDay) {
                lastDays.put(userId, curDay);
                daysActive.put(userId, daysActive.get(userId) + 1);
            }
        }
        final TIntDoubleMap averageDayDelta = new TIntDoubleHashMap();
        userSessions.forEachEntry((userId, nSessions) -> {
            averageDayDelta.put(userId, daysActive.get(userId) / ((double) nSessions) * DAY_HOURS);
            return true;
        });
        return averageDayDelta;
    }
}
