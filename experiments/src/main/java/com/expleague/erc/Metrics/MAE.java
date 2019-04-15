package com.expleague.erc.Metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Model;

import java.util.List;

public class MAE implements Metric {
    @Override
    public double calculate(List<Event> events, Model.Applicable applicable) {
        double errors = 0.;
        long count = 0;
        for (final Event event : events) {
            count++;
            final double expectedReturnTime = applicable.timeDelta(event.userId(), event.itemId());
            errors += Math.abs(event.getPrDelta() - expectedReturnTime);
            applicable.accept(event);
        }
        return errors / count;
    }
}
