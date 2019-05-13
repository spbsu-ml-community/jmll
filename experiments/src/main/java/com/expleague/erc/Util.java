package com.expleague.erc;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Util {
    public static final double MAX_GAP = .5;
    public static final int DAY_HOURS = 24;
    public static final int CHURN_THRESHOLD_DAYS = 2 * 7;
    public static final double CHURN_THRESHOLD = CHURN_THRESHOLD_DAYS * DAY_HOURS;

    public static boolean isDead(final double timeBetweenSessions) {
        return timeBetweenSessions > CHURN_THRESHOLD;
    }

    public static boolean isShortSession(final double timeBetweenSessions) {
        return timeBetweenSessions < MAX_GAP;
    }

    public static boolean forPrediction(final Session session) {
        return !isShortSession(session.getDelta()) && !isDead(session.getDelta());
    }

    public static double getDay(final double time, final int dayStart) {
        double lastBorder = ((int) time / DAY_HOURS) * DAY_HOURS + dayStart;
        if (lastBorder > time) {
            lastBorder -= DAY_HOURS;
        }
        return lastBorder;
    }

    public static int getDaysFromPrevSession(final Session session, final int dayStart) {
        return getDaysFromPrevSession(session.getDelta(), session.getStartTs(), dayStart);
    }

    public static int getDaysFromPrevSession(final double delta, final double time, final int dayStart) {
        double pos = Util.getDay(time, dayStart);
        double prevPos = Util.getDay(time - delta, dayStart);
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

    // Conversion for serialization

    public static Map<Integer, double[]> embeddingsToSerializable(final TIntObjectMap<Vec> embeddings) {
        return Arrays.stream(embeddings.keys())
                .boxed()
                .collect(Collectors.toMap(x -> x, x -> embeddings.get(x).toArray()));
    }

    public static TIntObjectMap<Vec> embeddingsFromSerializable(final Map<Integer, double[]> serMap) {
        final TIntObjectMap<Vec> embeddings = new TIntObjectHashMap<>();
        serMap.forEach((id, embedding) -> embeddings.put(id, new ArrayVec(embedding)));
        return embeddings;
    }

    public static Map<Integer, Double> intDoubleMapToSerializable(final TIntDoubleMap map) {
        return Arrays.stream(map.keys()).boxed().collect(Collectors.toMap(Function.identity(), map::get));
    }

    public static TIntDoubleMap intDoubleMapFromSerializable(final Map<Integer, Double> serializable) {
        final TIntDoubleMap map = new TIntDoubleHashMap();
        serializable.forEach(map::put);
        return map;
    }

    public static Map<Integer, Integer> intIntMapToSerializable(final TIntIntMap map) {
        return Arrays.stream(map.keys()).boxed().collect(Collectors.toMap(Function.identity(), map::get));
    }

    public static TIntIntMap intIntMapFromSerializable(final Map<Integer, Integer> serializable) {
        final TIntIntMap map = new TIntIntHashMap();
        serializable.forEach(map::put);
        return map;
    }

    public static class GetDelta implements TimeTransformer, Serializable {
        @Override
        public double apply(double delta, double time, int userId) {
            return delta;
        }
    }
}
