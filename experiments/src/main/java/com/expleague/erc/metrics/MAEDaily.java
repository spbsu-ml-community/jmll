package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.ModelDays;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.List;

public class MAEDaily implements Metric {
    private final TIntIntMap userDayBorders;

    public MAEDaily(List<Event> events) {
        userDayBorders = new TIntIntHashMap();
        ModelDays.calcDayPoints(events, userDayBorders, new TIntIntHashMap());
    }

    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        double errors = 0.;
        long count = 0;
        final TIntDoubleMap prevTimes = new TIntDoubleHashMap();
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int userId = session.userId();
            if (Util.forPrediction(session)) {
                final double actualReturnTime =
                        Util.getDaysFromPrevSession(session, userDayBorders.get(session.userId()));
                final double expectedReturnTime = applicable.timeDelta(userId, prevTimes.get(userId));
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            prevTimes.put(userId, session.getStartTs());
            applicable.accept(session);
        }
        return errors / count;
    }
}
