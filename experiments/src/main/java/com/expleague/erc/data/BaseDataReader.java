package com.expleague.erc.data;

import com.expleague.erc.Event;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public abstract class BaseDataReader {
    private Map<String, Integer> userMap = new HashMap<>();
    private Map<String, Integer> itemMap = new HashMap<>();

    @NotNull
    public List<Event> readData(final String dataPath, final int size) throws IOException {
        List<Event> events = Files.lines(Paths.get(dataPath))
                .limit(size)
                .map(this::makeEvent)
                .filter(Objects::nonNull)
                .sorted(Comparator.comparingDouble(Event::getTs))
                .collect(Collectors.toList());
        makePrDeltas(events);
        return events;
    }

    private void makePrDeltas(List<Event> events) {
        TIntObjectMap<TIntDoubleMap> lastTimeDone = new TIntObjectHashMap<>();
        for (Event event : events) {
            int uid = event.userId();
            int iid = event.itemId();
            if (!lastTimeDone.containsKey(uid)) {
                lastTimeDone.put(uid, new TIntDoubleHashMap());
            }
            if (lastTimeDone.get(uid).containsKey(iid)) {
                event.setPrDelta(event.getTs() - lastTimeDone.get(uid).get(iid));
            }
            lastTimeDone.get(uid).put(iid, event.getTs());
        }
    }

    protected abstract Event makeEvent(final String line);

    protected abstract double toTimestamp(final String timeString);

    protected int toUserId(final String userId) {
        if (!userMap.containsKey(userId)) {
            userMap.put(userId, userMap.size());
        }
        return userMap.get(userId);
    }

    protected int toItemId(final String itemId) {
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

    private Map<Integer, String> reverseMap(Map<String, Integer> map) {
        return map.keySet().stream().collect(Collectors.toMap(map::get, Function.identity()));
    }

    public Map<Integer, String> getReversedUserMap() {
        return reverseMap(userMap);
    }

    public Map<Integer, String> getReversedItemMap() {
        return reverseMap(itemMap);
    }
}
