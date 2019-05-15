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
            final double curTime = session.getStartTs();
            final double prevTime = prevTimes.get(userId);
            prevTimes.put(userId, curTime);
            final double actualReturnTime = session.getDelta();
            if (0 < actualReturnTime && !Util.isDead(actualReturnTime)) {
                final double expectedReturnTime = applicable.timeDelta(userId, prevTime);
                if (Double.isInfinite(Math.abs(actualReturnTime - expectedReturnTime))) {
                    continue;
                }
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            applicable.accept(session);
        }
        return errors / count;
    }
}
