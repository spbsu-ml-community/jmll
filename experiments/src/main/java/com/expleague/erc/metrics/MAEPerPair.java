package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Util;
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
        for (final EventSeq eventSeq : DataPreprocessor.groupToEventSeqs(events)) {
            final long pair = eventSeq.getPair();
            final double curTime = eventSeq.getStartTs();
            final double expectedReturnTime = applicable.timeDelta(eventSeq.userId(), eventSeq.itemId());
            final double prevTime = prevTimes.get(pair);
            final double actualReturnTime = curTime - prevTime;
            if (prevTime != prevTimes.getNoEntryValue() && !Util.isDead(actualReturnTime)) {
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            applicable.accept(eventSeq);
            prevTimes.put(pair, curTime);
        }
        return errors / count;
    }
}
