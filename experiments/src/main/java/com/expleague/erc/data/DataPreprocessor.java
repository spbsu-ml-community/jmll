package com.expleague.erc.data;

import com.expleague.erc.Event;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DataPreprocessor {
    public static class TrainTest {
        private List<Event> train;
        private List<Event> test;

        private TrainTest(List<Event> train, List<Event> test) {
            this.train = train;
            this.test = test;
        }

        public List<Event> getTrain() {
            return train;
        }

        public void setTrain(List<Event> events) {
            train = events;
        }

        public List<Event> getTest() {
            return test;
        }

        public void setTest(List<Event> events) {
            test = events;
        }

    }

    public TrainTest splitTrainTest(final List<Event> events, final double train_ratio) {
        double splitTime = getSplitTime(events, train_ratio);
        return new TrainTest(events.stream().filter(e -> e.getTs() <= splitTime).collect(Collectors.toList()),
                events.stream().filter(e -> e.getTs() > splitTime).collect(Collectors.toList()));
    }

    private double getSplitTime(final List<Event> events, final double percentile) {
        return events.stream().map(Event::getTs).sorted().collect(Collectors.toList()).get((int) (events.size() * percentile));
    }

    public TrainTest filter(final TrainTest trainTest, final int usersNum, final int itemsNum, final boolean isTop) {
        preFilter(trainTest, usersNum, itemsNum, isTop);
        List<Event> train = trainTest.getTrain();
        List<Event> test = trainTest.getTest();
        Set<String> users = intersect(toSet(train, Event::userId), toSet(test, Event::userId));
        Set<String> items = intersect(toSet(train, Event::itemId), toSet(test, Event::itemId));

        while (true) {
            train = filterByFieldValues(filterByFieldValues(trainTest.getTrain(), users, Event::userId), items, Event::itemId);
            test = filterByFieldValues(filterByFieldValues(trainTest.getTest(), users, Event::userId), items, Event::itemId);
            final Set<String> newUsers = intersect(toSet(train, Event::userId), toSet(test, Event::userId));
            final Set<String> newItems = intersect(toSet(train, Event::itemId), toSet(test, Event::itemId));
            if (users.equals(newUsers) && items.equals(newItems)) {
                break;
            }
            users = newUsers;
            items = newItems;
        }
        trainTest.setTrain(train);
        trainTest.setTest(test);
        return trainTest;
    }

    private void preFilter(final TrainTest trainTest, final int usersNum, final int itemsNum, final boolean isTop) {
        List<Event> events = trainTest.getTrain();
        events.addAll(trainTest.getTest());
        Set<String> users, items;
        if (isTop) {
            users = selectTopFields(events, usersNum, Event::userId);
            items = selectTopFields(events, itemsNum, Event::itemId);
        } else {
            users = selectRandomFields(events, usersNum, Event::userId);
            items = selectRandomFields(events, itemsNum, Event::itemId);
        }
        trainTest.setTrain(filterByFieldValues(filterByFieldValues(trainTest.getTrain(), users, Event::userId), items, Event::itemId));
        trainTest.setTest(filterByFieldValues(filterByFieldValues(trainTest.getTest(), users, Event::userId), items, Event::itemId));
    }

    private Set<String> selectRandomFields(final List<Event> events, final int size, final Function<Event, String> field) {
        List<String> fields = events.stream().map(field).distinct().collect(Collectors.toList());
        Collections.shuffle(fields);
        return new HashSet<>(fields.subList(0, size));
    }

    private Set<String> selectTopFields(final List<Event> events, final int size, final Function<Event, String> field) {
        Map<String, Integer> counts = new HashMap<>();
        events.stream().map(field).forEach(x -> counts.put(x, counts.getOrDefault(x, 0) + 1));
        List<Map.Entry<String, Integer>> entries = new ArrayList<>(counts.entrySet());
        entries.sort(Comparator.comparingInt(Map.Entry::getValue));
        return entries.stream().limit(size).map(Map.Entry::getKey).collect(Collectors.toSet());
    }

    private List<Event> filterByFieldValues(final List<Event> events, final Set<String> fieldValues, final Function<Event, String> field) {
        return events.stream().filter(e -> fieldValues.contains(field.apply(e))).collect(Collectors.toList());
    }

    private Set<String> toSet(final List<Event> events, final Function<Event, String> field) {
        return events.stream().map(field).collect(Collectors.toSet());
    }

    private Set<String> intersect(final Set<String> set1, final Set<String> set2) {
        Set<String> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        return intersection;
    }
}
