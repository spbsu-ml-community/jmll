package com.expleague.erc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.stream.Collectors;

public class Util {

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
