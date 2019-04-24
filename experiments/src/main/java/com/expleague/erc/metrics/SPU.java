package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.models.Model;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TLongDoubleHashMap;

import java.util.List;

public class SPU implements Metric {
    @Override
    public double calculate(List<Event> events, Model.Applicable applicable) {
        final TLongDoubleMap lastEventTimes = new TLongDoubleHashMap();
        final TLongDoubleMap predictedTimeDeltas = new TLongDoubleHashMap();
        double totalDiff = 0.;
        int count = 0;
        for (Event event: events) {
            final long pair = event.getPair();
            final double curEventTime = event.getTs();
            final double predictedTimeDelta = predictedTimeDeltas.get(pair);
            if (predictedTimeDelta != predictedTimeDeltas.getNoEntryValue()) {
                final double predictedEventSPU = 1 / predictedTimeDelta;
                final double realEventSPU = 1 / (curEventTime - lastEventTimes.get(pair));
                totalDiff += Math.abs(predictedEventSPU - realEventSPU);
                count++;
            }
            predictedTimeDeltas.put(pair, applicable.timeDelta(event.userId(), event.itemId()));
            lastEventTimes.put(pair, curEventTime);
            applicable.accept(event);
        }
        return totalDiff / count;
    }
}
