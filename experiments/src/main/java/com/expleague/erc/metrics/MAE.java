package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;

import java.util.List;

public class MAE implements Metric {
    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        double errors = 0.;
        long count = 0;
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final double expectedReturnTime = applicable.timeDelta(session.userId());
            final double actualReturnTime = session.getDelta();
            if (actualReturnTime > 0 && actualReturnTime < DataPreprocessor.CHURN_THRESHOLD) {
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            applicable.accept(session);
        }
        return errors / count;
    }
}
