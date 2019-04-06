package com.expleague.erc.data;

import com.expleague.erc.Event;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class LastFmDataReader {
    private Map<String, Integer> userMap = new HashMap<>();
    private Map<String, Integer> itemMap = new HashMap<>();

    @NotNull
    public List<Event> readData(final String dataPath, final int size) throws IOException {
        List<Event> events = Files.lines(Paths.get(dataPath))
                .limit(size)
                .map(this::makeEvent)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        Map<Integer, Map<Integer, Double>> lastTimeDone = new HashMap<>();
        events.sort(Comparator.comparingDouble(Event::getTs));
        for (Event event : events) {
            int uid = event.userId();
            int iid = event.itemId();
            if (!lastTimeDone.containsKey(uid)) {
                lastTimeDone.put(uid, new HashMap<>());
            }
            if (lastTimeDone.get(uid).containsKey(iid)) {
                event.setPrDelta(event.getTs() - lastTimeDone.get(uid).get(iid));
            }
            lastTimeDone.get(uid).put(iid, event.getTs());
        }
        return events;
    }

    private Event makeEvent(final String line) {
        String[] words = line.split("\t");
        try {
            return new Event(toUserId(words[1]), toItemId(words[4]), toTimestamp(words[2]));
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    private double toTimestamp(final String timeString) {
        long time = javax.xml.bind.DatatypeConverter.parseDateTime(timeString).getTimeInMillis();
        double seconds = (double) time / 1000;
        return seconds / (60 * 60);
    }

    private int toUserId(final String userId) {
        if (!userMap.containsKey(userId)) {
            userMap.put(userId, userMap.size());
        }
        return userMap.get(userId);
    }

    private int toItemId(final String itemId) {
        if (!itemMap.containsKey(itemId)) {
            itemMap.put(itemId, itemMap.size());
        }
        return itemMap.get(itemId);
    }

    public Map<String, Integer> getUserMap() {
        return userMap;
    }

    public Map<String, Integer> getItemMap() {
        return itemMap;
    }
}
