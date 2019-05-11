package com.expleague.erc.models;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

import static com.expleague.erc.Util.DAY_HOURS;

public class ModelCombined extends Model {

    private TIntIntMap userDayBorders = new TIntIntHashMap();
    private TIntIntMap userDayPeaks = new TIntIntHashMap();
    private TIntDoubleMap userDayAvgStarts;

    private Model daysModel;
    private Model timeModel;

    public ModelCombined(int dim, double beta, double eps, double otherItemImportance,
                         DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                         LambdaStrategyFactory lambdaStrategyFactory) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
        daysModel = new ModelExpPerUser(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform,
                lambdaStrategyFactory, null, // TODO: fix initializing of initLambdas
                (x, userId) -> Util.getDay(x, userDayBorders.get(userId)), x -> true, x -> x < 14);
        timeModel = new ConstantNextTimeModel(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
    }

    @Override
    public void initModel(final List<Event> events) {
        if (isInit) {
            return;
        }
        daysModel.makeInitialEmbeddings(events);
        timeModel.makeInitialEmbeddings(events);
        initIds();
        userDayAvgStarts = calcAvgStarts(events);
        calcDayPoints(events, userDayBorders, userDayPeaks);
        isInit = true;
    }

    @Override
    public void fit(final List<Event> events, final double learningRate, final int iterationsNumber,
                    final double decay, final FitListener listener) {
        daysModel.fit(events, learningRate, iterationsNumber, decay, listener);
        timeModel.fit(events, learningRate, iterationsNumber, decay, listener);
    }

    private class ApplicableImpl implements ApplicableModel {
        private final ApplicableModel daysApplicable;
        private final ApplicableModel timeApplicable;

        private ApplicableImpl(final ApplicableModel daysApplicable, final ApplicableModel timeApplicable) {
            this.daysApplicable = daysApplicable;
            this.timeApplicable = timeApplicable;
        }

        @Override
        public void accept(final EventSeq eventSeq) {
            daysApplicable.accept(eventSeq);
            timeApplicable.accept(eventSeq);
        }

        @Override
        public double timeDelta(final int userId, final double time) {
            final double dayPrediction = daysApplicable.timeDelta(userId, time);
            if (time + dayPrediction * DAY_HOURS < Util.getDay(time, userDayBorders.get(userId)) + DAY_HOURS) {
                return timeApplicable.timeDelta(userId, time);
            }
            final int predictedDay = (int) (time + (int) dayPrediction * DAY_HOURS) / DAY_HOURS;
            final double predictedTime = predictedDay * DAY_HOURS + userDayAvgStarts.get(userId);
            return predictedTime - time;
        }
    }

    @Override
    public ApplicableModel getApplicable() {
        return new ApplicableImpl(daysModel.getApplicable(), timeModel.getApplicable());
    }

    public static void calcDayPoints(final List<Event> events, final TIntIntMap userDayBorders, final TIntIntMap userDayPeaks) {
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
                if (userCounters[i] > maxCounter) {
                    maxCounter = userCounters[i];
                    argMax = i;
                }
            }
            userDayBorders.put(userId, argMin);
            userDayPeaks.put(userId, argMax);
            return true;
        });
    }

    private static TIntDoubleMap calcAvgStarts(final List<Event> events) {
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
}
