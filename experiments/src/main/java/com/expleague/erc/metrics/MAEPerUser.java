package com.expleague.erc.metrics;

import com.expleague.erc.Event;
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
        for (final Event event : events) {
            final int userId = event.userId();
            final double curTime = event.getTs();
            final double expectedReturnTime = applicable.timeDelta(userId, event.itemId());
            final double prevTime = prevTimes.get(userId);
            if (prevTime != prevTimes.getNoEntryValue()) {
                final double actualReturnTime = curTime - prevTime;
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            applicable.accept(event);
            prevTimes.put(userId, curTime);
        }
        return errors / count;
    }
}
