package com.expleague.erc.metrics;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.Model;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TLongDoubleHashMap;
import gnu.trove.set.TLongSet;
import gnu.trove.set.hash.TLongHashSet;

import java.util.List;

import static java.lang.Math.*;

public class LogLikelihood implements Metric {
    private final double eps;

    public LogLikelihood(double eps) {
        this.eps = eps;
    }

    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        final double observationEnd = events.get(events.size() - 1).getTs();
        double logLikelihood = 0.;
        final TLongSet seenPairs = new TLongHashSet();
        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
        for (final EventSeq eventSeq : DataPreprocessor.groupToEventSeqs(events)) {
            final int userId = eventSeq.userId();
            final int itemId = eventSeq.itemId();
            final long pairId = eventSeq.getPair();
            if (!seenPairs.contains(pairId)) {
                seenPairs.add(pairId);
                applicable.accept(eventSeq);
            } else {
                final double prDelta = max(eventSeq.getDelta(), eps);
                final double p = applicable.probabilityInterval(userId, itemId, prDelta - eps, prDelta + eps);
                assert 0 <= p && p <= 1;
                if (p > 0) {
                    logLikelihood += log(p);
                }
                applicable.accept(eventSeq);
                lastVisitTimes.put(pairId, eventSeq.getStartTs());
            }
        }
//        for (long pairId : lastVisitTimes.keys()) {
//            final int userId = Util.extractUserId(pairId);
//            final int itemId = Util.extractItemId(pairId);
//            final double tau = observationEnd - lastVisitTimes.get(pairId);
//            if (tau > 0) {
//                logLikelihood += log(1 - applicable.probabilityBeforeX(userId, itemId, tau));
//            }
//        }
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
