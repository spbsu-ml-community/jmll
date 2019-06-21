package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;

import java.util.List;

public class SPUByDelta implements Metric {
    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        final TIntDoubleMap predictedTimeDeltas = new TIntDoubleHashMap();
        double totalDiff = 0.;
        int count = 0;
        for (final EventSeq eventSeq: DataPreprocessor.groupToEventSeqs(events)) {
            final int user = eventSeq.userId();
            final double predictedTimeDelta = predictedTimeDeltas.get(user);
            final double realEventTimeDelta = eventSeq.getDelta();
            if (predictedTimeDelta != predictedTimeDeltas.getNoEntryValue() && realEventTimeDelta > 0. &&
                    !Util.isDead(realEventTimeDelta)) {
                final double predictedEventSPU = 1 / predictedTimeDelta;
                final double realEventSPU = 1 / realEventTimeDelta;
                totalDiff += Math.abs(predictedEventSPU - realEventSPU);
                count++;
            }
            predictedTimeDeltas.put(user, applicable.timeDelta(eventSeq.userId(), eventSeq.itemId()));
            applicable.accept(eventSeq);
        }
        return totalDiff / count;
    }
}
