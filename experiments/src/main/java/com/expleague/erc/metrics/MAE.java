package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.models.Model;

import java.util.List;

public class MAE implements Metric {
    @Override
    public double calculate(List<Event> events, Model.Applicable applicable) {
        double errors = 0.;
        long count = 0;
        for (final Event event : events) {
            final double expectedReturnTime = applicable.timeDelta(event.userId(), event.itemId());
            final double actualReturnTime = event.getPrDelta();
            if (actualReturnTime > 0) {
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            applicable.accept(event);
        }
        return errors / count;
    }
}
