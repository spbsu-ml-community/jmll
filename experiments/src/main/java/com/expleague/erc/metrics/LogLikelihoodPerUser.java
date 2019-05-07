package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;

import java.util.List;

import static java.lang.Math.log;
import static java.lang.Math.max;

public class LogLikelihoodPerUser implements Metric {
    private final double eps;

    public LogLikelihoodPerUser(double eps) {
        this.eps = eps;
    }

    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        double logLikelihood = 0.;
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final double delta = max(session.getDelta(), eps);
            if (0 < delta && delta < DataPreprocessor.CHURN_THRESHOLD) {
                final double p = applicable.probabilityInterval(session.userId(), delta - eps, delta + eps);
                assert 0 <= p && p <= 1;
                if (p > 0) {
                    logLikelihood += log(p);
                }
            }
            applicable.accept(session);
        }
        return logLikelihood;
    }
}
