package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;

import java.util.List;

public class MAEPerUser implements Metric {
    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        double errors = 0.;
        long count = 0;
        final TIntDoubleMap prevTimes = new TIntDoubleHashMap();
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int userId = session.userId();
            if (Util.forPrediction(session)) {
                final double expectedReturnTime = applicable.timeDelta(userId, prevTimes.get(userId));
                count++;
                errors += Math.abs(session.getDelta() - expectedReturnTime);
            }
            prevTimes.put(userId, session.getStartTs());
            applicable.accept(session);
        }
        return errors / count;
    }
}
