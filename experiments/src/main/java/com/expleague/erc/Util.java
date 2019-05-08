package com.expleague.erc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.stream.Collectors;

public class Util {
    public static final double CHURN_THRESHOLD = 2 * 7 * 24.;
    public static final double MAX_GAP = .5;
    private static final int DAY_HOURS = 24;

    public static boolean isDead(final double timeBetweenSessions) {
        return timeBetweenSessions > CHURN_THRESHOLD;
    }

    public static boolean isShortSession(final double timeBetweenSessions) {
        return timeBetweenSessions < MAX_GAP;
    }

    public static double getDay(final double time, final int dayStart) {
        double lastBorder = ((int) time / DAY_HOURS) * DAY_HOURS + dayStart;
        if (lastBorder > time) {
            lastBorder -= DAY_HOURS;
        }
        return lastBorder;
    }

    public static int getDaysFromPrevSession(final Session session, final int dayStart) {
        double pos = Util.getDay(session.getStartTs(), dayStart);
        double prevPos = Util.getDay(session.getStartTs() - session.getDelta(), dayStart);
        return (int) (pos - prevPos) / DAY_HOURS;
    }

    public static int extractUserId(final long idPair) {
        return (int)(idPair >> 32);
    }

    public static long combineIds(final int userId, final int itemId) {
        return (long) userId << 32 | itemId;
    }

    public static int extractItemId(final long idPair) {
        return (int)idPair;
    }

    public static void writeMap(Path filePath, Map<Integer, String> map) throws IOException {
        final String strRep = map.keySet().stream()
                .map(key -> key + "\t" + map.get(key))
                .collect(Collectors.joining("\n", "", "\n"));
        Files.write(filePath, strRep.getBytes(), StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.CREATE);
    }
}
