package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TLongDoubleHashMap;

import java.util.List;

public class SPU implements Metric {
    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        final TLongDoubleMap lastEventTimes = new TLongDoubleHashMap();
        final TLongDoubleMap predictedTimeDeltas = new TLongDoubleHashMap();
        double totalDiff = 0.;
        int count = 0;
        for (final EventSeq eventSeq : DataPreprocessor.groupToEventSeqs(events)) {
            final long pair = eventSeq.getPair();
            final double curEventTime = eventSeq.getTs();
            final double predictedTimeDelta = predictedTimeDeltas.get(pair);
            if (predictedTimeDelta != predictedTimeDeltas.getNoEntryValue()) {
                final double predictedEventSPU = 1 / predictedTimeDelta;
                final double realEventSPU = 1 / (curEventTime - lastEventTimes.get(pair));
                totalDiff += Math.abs(predictedEventSPU - realEventSPU);
                count++;
            }
            predictedTimeDeltas.put(pair, applicable.timeDelta(eventSeq.userId(), eventSeq.itemId()));
            lastEventTimes.put(pair, curEventTime);
            applicable.accept(eventSeq);
        }
        return totalDiff / count;
    }
}
