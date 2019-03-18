package com.expleague.erc.data;

import com.expleague.erc.Event;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.util.*;
import java.util.stream.Collectors;

public class LastFmDataReader {
    @NotNull
    public List<Event> readData(final String dataPath, final int size) throws IOException {
        List<Event> events = Files.lines(Paths.get(dataPath))
                .limit(size)
                .map(this::makeEvent)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        Map<String, Map<String, Double>> lastTimeDone = new HashMap<>();
        events.sort(Comparator.comparingDouble(Event::getTs));
        for (Event event : events) {
            String uid = event.userId();
            String iid = event.itemId();
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
            return new Event(words[1], words[4], toTimestamp(words[2]));
        } catch (ParseException e) {
            return null;
        }
    }

    private double toTimestamp(final String timeString) throws ParseException {
        long time = javax.xml.bind.DatatypeConverter.parseDateTime(timeString).getTimeInMillis();
        double seconds = (double) time / 1000;
        return seconds / (60 * 60);
    }
}
