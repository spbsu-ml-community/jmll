package com.expleague.erc.Metrics;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;
import com.expleague.erc.Model;
import com.expleague.erc.Util;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TLongDoubleHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

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
        final TIntObjectMap<TIntSet> seenItems = new TIntObjectHashMap<>();
        for (final int userId : events.stream().mapToInt(Event::userId).toArray()) {
            seenItems.put(userId, new TIntHashSet());
        }
        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
        for (final Event event : events) {
            final int userId = event.userId();
            final int itemId = event.itemId();
            if (!seenItems.get(userId).contains(itemId)) {
                seenItems.get(userId).add(itemId);
                applicable.accept(event);
            } else {
                final double lambda = applicable.getLambda(userId, itemId);
                final double prDelta = event.getPrDelta();
                final double spanBegin, spanEnd;
                if (prDelta - eps < 0) {
                    spanBegin = 0;
                    spanEnd = 2 * eps;
                } else {
                    spanBegin = prDelta - eps;
                    spanEnd = prDelta + eps;
                }
                double probability = -Math.exp(-lambda * spanEnd) + Math.exp(-lambda * spanBegin);
                final double logLikelihoodDelta =
                        Math.log(probability);
                    logLikelihood += logLikelihoodDelta;

                applicable.accept(event);
            }
            lastVisitTimes.put(Util.combineIds(userId, itemId), event.getTs());
        }

//        final double observationEnd = events.get(events.size() - 1).getTs();
//        double logLikelihood = 0.;
//        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap(128, .5f, -1, -1);
//        for (final Event event : events) {
//            final int userId = event.userId();
//            final int itemId = event.itemId();
//            final long pair = event.getPair();
//            if (lastVisitTimes.get(pair) != lastVisitTimes.getNoEntryValue()) {
//                final double lambda = applicable.getLambda(userId, itemId);
//                final double prDelta = event.getPrDelta();
//                final double logLikelihoodDelta =
//                        Math.log(-Math.exp(-lambda * (prDelta + eps)) +
//                                Math.exp(-lambda * Math.max(0, prDelta - eps)));
//                if (Double.isFinite(logLikelihoodDelta)) {
//                    logLikelihood += logLikelihoodDelta;
//                }
//            }
//            lastVisitTimes.put(pair, event.getTs());
//            applicable.accept(event);
//        }
        for (long pairId : lastVisitTimes.keys()) {
            final int userId = (int)(pairId >> 32);
            final int itemId = (int)pairId;
            final double lambda = applicable.getLambda(userId, itemId);
            final double logLikelihoodDelta = -lambda * (observationEnd - lastVisitTimes.get(pairId));
            if (Double.isFinite(logLikelihoodDelta)) {
                logLikelihood += logLikelihoodDelta;
            }
        }
        return logLikelihood;
    }

//    TODO: find a better place
    public static void differentiate(TIntObjectMap<Vec> parameters, TIntObjectMap<Vec> target,
                                     Model model, LogLikelihood llCalculator,
                                     List<Event> prepEvents, List<Event> diffEvents, double step) {
        final int dim = model.getDim();
        final double initialLL = llCalculator.calculate(diffEvents, model.getApplicable(prepEvents));
        parameters.forEachEntry((key, embedding) -> {
            final ArrayVec curDerivative = new ArrayVec(dim);
            for (int i = 0; i < dim; ++i) {
                final double oldVal = embedding.get(i);
                embedding.set(i, oldVal + step);
                final double curLL = llCalculator.calculate(diffEvents, model.getApplicable(prepEvents));
                curDerivative.set(i, (curLL - initialLL) / step);
                embedding.set(i, oldVal);
            }
            target.put(key, curDerivative);
            return true;
        });
    }
}
