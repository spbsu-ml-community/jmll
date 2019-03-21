package com.expleague.erc.data;

import com.expleague.commons.util.Pair;
import com.expleague.erc.Event;

import java.util.*;

public class TSRDataProcessor extends DataPreprocessor {
    public TrainTest splitTrainTest(final List<Event> events, final double train_ratio) {
        Map<Pair<Integer, Integer>, List<Event>> eventsByPair = new HashMap<>();
        for (final Event event : events) {
            Pair<Integer, Integer> pair = new Pair<>(event.userId(), event.itemId());
            if (!eventsByPair.containsKey(pair)) {
                eventsByPair.put(pair, new ArrayList<>());
            }
            eventsByPair.get(pair).add(event);
        }
        final List<Event> train = new ArrayList<>();
        final List<Event> test = new ArrayList<>();
        for (Pair<Integer, Integer> pair : eventsByPair.keySet()) {
            List<Event> pairEvents = eventsByPair.get(pair);
            pairEvents.sort(Comparator.comparingDouble(Event::getTs));
            int trainSize = (int) (pairEvents.size() * train_ratio);
            if (trainSize < 2) {
                continue;
            }
            for (int i = 0; i < trainSize; i++) {
                train.add(pairEvents.get(i));
            }
            for (int i = trainSize; i < pairEvents.size(); i++) {
                test.add(pairEvents.get(i));
            }
        }
        train.sort(Comparator.comparingDouble(Event::getTs));
        test.sort(Comparator.comparingDouble(Event::getTs));
        return new TrainTest(train, test);
    }


}
