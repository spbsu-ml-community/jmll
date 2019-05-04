package com.expleague.erc.metrics;

import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;

import java.util.List;

public class MAE implements Metric {
    @Override
    public double calculate(List<Event> events, ApplicableModel applicable) {
        double errors = 0.;
        long count = 0;
        for (final EventSeq eventSeq : DataPreprocessor.groupToEventSeqs(events)) {
            final double expectedReturnTime = applicable.timeDelta(eventSeq.userId(), eventSeq.itemId());
            final double actualReturnTime = eventSeq.getDelta();
            if (actualReturnTime > 0 && actualReturnTime < DataPreprocessor.CHURN_THRESHOLD) {
                count++;
                errors += Math.abs(actualReturnTime - expectedReturnTime);
            }
            applicable.accept(eventSeq);
        }
        return errors / count;
    }
}
