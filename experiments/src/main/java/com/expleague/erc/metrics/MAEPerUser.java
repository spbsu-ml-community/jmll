package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
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
        for (final EventSeq eventSeq : DataPreprocessor.groupToEventSeqs(events)) {
            final int userId = eventSeq.userId();
            final double curTime = eventSeq.getTs();
            final double expectedReturnTime = applicable.timeDelta(userId, eventSeq.itemId());
            final double prevTime = prevTimes.get(userId);
            final double actualReturnTime = curTime - prevTime;
            if (prevTime != prevTimes.getNoEntryValue() && actualReturnTime < DataPreprocessor.CHURN_THRESHOLD) {
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            applicable.accept(eventSeq);
            prevTimes.put(userId, curTime);
        }
        return errors / count;
    }
}
