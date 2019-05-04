package com.expleague.erc.data;

import com.expleague.erc.Event;

public class TolokaDataReader extends BaseDataReader {
    protected Event makeEvent(final String line) {
        String[] words = line.split("\t");
        try {
            return new Event(toUserId(words[0]), toItemId(words[1]), toTimestamp(words[2]));
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    protected double toTimestamp(final String timeString) {
        return Double.valueOf(timeString);
    }
}
