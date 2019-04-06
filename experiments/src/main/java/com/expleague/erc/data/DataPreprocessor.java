package com.expleague.erc.data;

import com.expleague.erc.Event;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public abstract class DataPreprocessor {
    public static class TrainTest {
        private List<Event> train;
        private List<Event> test;

        protected TrainTest(List<Event> train, List<Event> test) {
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

    public abstract TrainTest splitTrainTest(final List<Event> events, final double train_ratio);

    public TrainTest filter(final TrainTest trainTest, final int usersNum, final int itemsNum, final boolean isTop) {
        preFilter(trainTest, usersNum, itemsNum, isTop);
        List<Event> train = trainTest.getTrain();
        List<Event> test = trainTest.getTest();
        Set<Integer> users = intersect(toSet(train, Event::userId), toSet(test, Event::userId));
        Set<Integer> items = intersect(toSet(train, Event::itemId), toSet(test, Event::itemId));

        while (true) {
            train = filterByFieldValues(filterByFieldValues(trainTest.getTrain(), users, Event::userId), items, Event::itemId);
            test = filterByFieldValues(filterByFieldValues(trainTest.getTest(), users, Event::userId), items, Event::itemId);
            final Set<Integer> newUsers = intersect(toSet(train, Event::userId), toSet(test, Event::userId));
            final Set<Integer> newItems = intersect(toSet(train, Event::itemId), toSet(test, Event::itemId));
            if (users.equals(newUsers) && items.equals(newItems)) {
                break;
            }
            users = newUsers;
            items = newItems;
        }

        train.sort(Comparator.comparingDouble(Event::getTs));
        test.sort(Comparator.comparingDouble(Event::getTs));
        trainTest.setTrain(train);
        trainTest.setTest(test);
        System.out.println("|Train| = " + train.size() + ", |Test| = " + test.size());
        System.out.println("|Users| = " + users.size() + ", |Items| = " + items.size());
        return trainTest;
    }

    private void preFilter(final TrainTest trainTest, final int usersNum, final int itemsNum, final boolean isTop) {
        List<Event> events = new ArrayList<>(trainTest.getTrain());
        events.addAll(trainTest.getTest());
        Set<Integer> users, items;
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

    private Set<Integer> selectRandomFields(final List<Event> events, final int size, final Function<Event, Integer> field) {
        List<Integer> fields = events.stream().map(field).distinct().collect(Collectors.toList());
        Collections.shuffle(fields);
        return new HashSet<>(fields.subList(0, Math.min(fields.size(), size)));
    }

    private Set<Integer> selectTopFields(final List<Event> events, final int size, final Function<Event, Integer> field) {
        Map<Integer, Integer> counts = new HashMap<>();
        events.stream().map(field).forEach(x -> counts.put(x, counts.getOrDefault(x, 0) + 1));
        List<Map.Entry<Integer, Integer>> entries = new ArrayList<>(counts.entrySet());
        entries.sort(Comparator.comparingInt(x -> -x.getValue()));
        return entries.stream().limit(size).map(Map.Entry::getKey).collect(Collectors.toSet());
    }

    private List<Event> filterByFieldValues(final List<Event> events, final Set<Integer> fieldValues, final Function<Event, Integer> field) {
        return events.stream().filter(e -> fieldValues.contains(field.apply(e))).collect(Collectors.toList());
    }

    protected Set<Integer> toSet(final List<Event> events, final Function<Event, Integer> field) {
        return events.stream().map(field).collect(Collectors.toSet());
    }

    private <T> Set<T> intersect(final Set<T> set1, final Set<T> set2) {
        Set<T> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        return intersection;
    }
}
