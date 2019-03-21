package com.expleague.erc.data;

import com.expleague.erc.Event;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class OneTimeDataProcessor extends DataPreprocessor {
    public TrainTest splitTrainTest(final List<Event> events, final double train_ratio) {
        double splitTime = getSplitTime(events, train_ratio);
        final List<Event> train = events.stream().filter(e -> e.getTs() <= splitTime)
                .sorted(Comparator.comparingDouble(Event::getTs)).collect(Collectors.toList());
        final List<Event> test = events.stream().filter(e -> e.getTs() > splitTime)
                .sorted(Comparator.comparingDouble(Event::getTs)).collect(Collectors.toList());
        return new TrainTest(train, test);
    }

    private double getSplitTime(final List<Event> events, final double percentile) {
        return events.stream().map(Event::getTs).sorted().collect(Collectors.toList()).get((int) (events.size() * percentile));
    }
}
