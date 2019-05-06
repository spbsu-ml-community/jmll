package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;

import java.util.List;

public class SPUPerUser implements Metric {
    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        final TIntDoubleMap lastEventTimes = new TIntDoubleHashMap();
        final TIntDoubleMap predictedTimeDeltas = new TIntDoubleHashMap();
        double totalDiff = 0.;
        int count = 0;
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int user = session.userId();
            final double curEventTime = session.getStartTs();
            final double predictedTimeDelta = predictedTimeDeltas.get(user);
            final double userLastEventTime = lastEventTimes.get(user);
            final double realEventTimeDelta = curEventTime - userLastEventTime;
            if (predictedTimeDelta != predictedTimeDeltas.getNoEntryValue() && realEventTimeDelta != 0.
                    && session.getDelta() < DataPreprocessor.CHURN_THRESHOLD) {
                final double predictedEventSPU = 1 / predictedTimeDelta;
                final double realEventSPU = 1 / realEventTimeDelta;
                totalDiff += Math.abs(predictedEventSPU - realEventSPU);
                count++;
            }
            predictedTimeDeltas.put(user, applicable.timeDelta(user, userLastEventTime));
            lastEventTimes.put(user, curEventTime);
            applicable.accept(session);
        }
        return totalDiff / count;
    }
}
