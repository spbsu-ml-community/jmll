package com.expleague.erc.data;

import com.expleague.erc.Event;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public class LastFmDataReader {
    @NotNull
    public List<Event> readData(final String dataPath, final int size) throws IOException {
        return Files.lines(Paths.get(dataPath))
                .limit(size)
                .map(this::makeEvent)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
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
        DateFormat formatter = new SimpleDateFormat("yyyy-MM-ddTHH:mm:ssZ");
        long time = formatter.parse(timeString).getTime();
        double seconds = (double) time / 1000;
        return seconds / (60 * 60);
    }
}
