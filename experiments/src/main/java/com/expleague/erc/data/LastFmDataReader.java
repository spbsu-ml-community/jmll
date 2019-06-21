package com.expleague.erc.data;

import com.expleague.erc.Event;

public class LastFmDataReader extends BaseDataReader {
    protected Event makeEvent(final String line) {
        String[] words = line.split("\t");
        try {
            return new Event(toUserId(words[0]), toItemId(words[3]), toTimestamp(words[1]));
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    protected double toTimestamp(final String timeString) {
        long time = javax.xml.bind.DatatypeConverter.parseDateTime(timeString).getTimeInMillis();
        double seconds = (double) time / 1000;
        return seconds / (60 * 60);
    }
}
