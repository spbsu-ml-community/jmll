package com.expleague.joom.tags;

import com.expleague.commons.csv.CsvTools;
import com.expleague.commons.util.logging.Interval;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.zip.GZIPInputStream;

public class Freqs {
  public static final String WD = ".";

  public static void main(String[] args) throws IOException {
    TObjectIntHashMap<String> tagFreqs = new TObjectIntHashMap<>();
    try (final Reader reader = new InputStreamReader(new GZIPInputStream(Files.newInputStream(Paths.get(WD + "/search_sessions_cats.csv.gz"))), StandardCharsets.UTF_8)) {
      final long[] counter = new long[]{0};
      Interval.start();
      TDoubleArrayList times = new TDoubleArrayList();
      CsvTools.csvLines(reader, ',', '"', '\\', true).forEach(row -> {
        if (++counter[0] % 1000000 == 0) {
          times.add(Interval.time());
          times.sort();
          System.out.print("\r" + counter[0] + " lines processed for: " + Interval.time() + " median: " + times.get(times.size() / 2));
          Interval.start();
        }
        final String tag = row.asString("cat");
        if (tag != null)
          tagFreqs.adjustOrPutValue(tag, 1, 1);
      });
    }
    try (final Writer out = Files.newBufferedWriter(Paths.get(WD + "/cat-freqs.txt"))) {
      out.append("tag,freq\n");
      final List<String> tagsLst = new ArrayList<>(tagFreqs.keySet());
      tagsLst.sort(Comparator.comparingLong(tag -> -tagFreqs.get(tag)));
      for (String tag : tagsLst) {
        out.append('"').append(tag.replace("\"", "\"\"")).append('"').append(',').append(Integer.toString(tagFreqs.get(tag))).append('\n');
      }
    }
  }
}
