package com.expleague.erc.Metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Model;
import com.expleague.erc.Util;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TLongDoubleHashMap;

import java.util.List;

public class LogLikelihood implements Metric {
    private final double eps;

    public LogLikelihood(double eps) {
        this.eps = eps;
    }

    @Override
    public double calculate(List<Event> events, Model.Applicable applicable) {
        final double observationEnd = events.get(events.size() - 1).getTs();
        double logLikelihood = 0.;
        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
        for (final Event event : events) {
            final int userId = event.userId();
            final int itemId = event.itemId();
            if (!event.isFinish()) {
                final double lambda = applicable.getLambda(userId, itemId);
                double prDelta = event.getPrDelta();
                final double logLikelihoodDelta =
                        Math.log(-Math.exp(-lambda * (prDelta + eps)) +
                                Math.exp(-lambda * Math.max(0, prDelta - eps)));
                if (Double.isFinite(logLikelihoodDelta)) {
                    logLikelihood += logLikelihoodDelta;
                }
                lastVisitTimes.put(Util.combineIds(userId, itemId), event.getTs());
            }
            applicable.accept(event);
        }
        for (long pairId : lastVisitTimes.keys()) {
            final int userId = (int)(pairId >> 32);
            final int itemId = (int)pairId;
            final double lambda = applicable.getLambda(userId, itemId);
            final double logLikelihoodDelta = -lambda * observationEnd - lastVisitTimes.get(pairId);
            if (Double.isFinite(logLikelihoodDelta)) {
                logLikelihood += logLikelihoodDelta;
            }
        }
        return logLikelihood;
    }
}
