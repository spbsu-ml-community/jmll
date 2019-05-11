package com.expleague.erc.models;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

import static com.expleague.erc.Util.DAY_HOURS;

public class ConstantNextTimeModel extends Model {

    private TIntDoubleMap averageOneDayDelta;

    public ConstantNextTimeModel(int dim, double beta, double eps, double otherItemImportance,
                                 DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                                 LambdaStrategyFactory lambdaStrategyFactory) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
    }

    @Override
    public void initModel(final List<Event> events) {
        makeInitialEmbeddings(events);
        initIds();
        averageOneDayDelta = calcAverageOneDayDelta(events);
        isInit = true;
    }

    private class ApplicableImpl implements ApplicableModel {
        private ApplicableImpl() {

        }

        @Override
        public void accept(final EventSeq eventSeq) {

        }

        @Override
        public double timeDelta(final int userId, final double time) {
            return averageOneDayDelta.get(userId);
        }
    }

    public ApplicableModel getApplicable() {
        return new ApplicableImpl();
    }

    private static TIntDoubleMap calcAverageOneDayDelta(final List<Event> events) {
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
}
