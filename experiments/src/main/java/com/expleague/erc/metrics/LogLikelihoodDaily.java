package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.ModelCombined;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.List;

import static java.lang.Math.log;

public class LogLikelihoodDaily implements Metric {
    private final TIntIntMap userDayBorders;

    public LogLikelihoodDaily(List<Event> events) {
        userDayBorders = ModelCombined.findMinHourInDay(events);
    }

    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        double logLikelihood = 0.;
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            if (Util.forPrediction(session)) {
                final int daysPassed = Util.getDaysFromPrevSession(session, userDayBorders.get(session.userId()));
                final double p = applicable.probabilityInterval(session.userId(), daysPassed, daysPassed + 1);
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
