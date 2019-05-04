package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TLongDoubleHashMap;

import java.util.List;

public class MAEPerPair implements Metric {
    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        double errors = 0.;
        long count = 0;
        final TLongDoubleMap prevTimes = new TLongDoubleHashMap();
        for (final Session session : DataPreprocessor.groupToSessions(events)) {
            final long pair = session.getPair();
            final double curTime = session.getTs();
            final double expectedReturnTime = applicable.timeDelta(session.userId(), session.itemId());
            final double prevTime = prevTimes.get(pair);
            final double actualReturnTime = curTime - prevTime;
            if (prevTime != prevTimes.getNoEntryValue() && actualReturnTime < DataPreprocessor.CHURN_THRESHOLD) {
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            applicable.accept(session);
            prevTimes.put(pair, curTime);
        }
        return errors / count;
    }
}
