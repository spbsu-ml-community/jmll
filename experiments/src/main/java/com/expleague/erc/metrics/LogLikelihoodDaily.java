package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.ModelDays;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.List;

import static java.lang.Math.log;
import static java.lang.Math.min;

public class LogLikelihoodDaily implements Metric {
    private final TIntIntMap userDayBorders;
    private final double eps;

    public LogLikelihoodDaily(List<Event> events, double eps) {
        this.eps = eps;
        userDayBorders = new TIntIntHashMap();
        ModelDays.calcDayPoints(events, userDayBorders, new TIntIntHashMap());
    }

    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        double logLikelihood = 0.;
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            if (Util.forPrediction(session)) {
                final int daysPassed = Util.getDaysFromPrevSession(session, userDayBorders.get(session.userId()));
                final double tau = min(daysPassed, eps);
                final double p = applicable.probabilityInterval(session.userId(), tau - eps, tau + eps);
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
