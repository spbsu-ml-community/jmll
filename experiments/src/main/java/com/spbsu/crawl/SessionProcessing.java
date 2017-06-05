package com.spbsu.crawl;

import com.spbsu.commons.func.Processor;
import com.spbsu.commons.func.types.impl.TypeConvertersCollection;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.meta.DSItem;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Experts League
 * Created by solar on 28.07.16.
 */
public class SessionProcessing {
  public static void main(String[] args) throws IOException {
    final File root = new File(".");
    final List<GameState> items = new ArrayList<>();
    final VecBuilder target = new VecBuilder();
    final VecBuilder data = new VecBuilder();
    final int[] featuresCount = new int[]{-1};
    final String[] sessions = root.list((ctxt, name) -> name.startsWith("session") && name.endsWith(".txt"));
    assert sessions != null;
    for (final String session : sessions) {
      final File sessionFile = new File(root, session);
      final String sessionName = sessionFile.getName().replace(".txt", "");
      CharSeqTools.processLines(new InputStreamReader(new FileInputStream(sessionFile), StreamTools.UTF), new Processor<CharSequence>() {
        int lindex = 0;

        @Override
        public void process(final CharSequence arg) {
          lindex++;
          final Vec point = TypeConvertersCollection.ROOT.convert(arg, Vec.class);
          final int validFeatures = point.dim() - 1;
          items.add(new GameState(sessionName, lindex));
          if (lindex > 0)
            target.append(0);
          if (featuresCount[0] < 0)
            featuresCount[0] = validFeatures;
          else if (featuresCount[0] != validFeatures)
            throw new RuntimeException("Failed to parse line: " + lindex);
          for (int i = 0; i < validFeatures; i++) {
            data.append(point.get(i));
          }
        }
      });
      target.append(1);

    }
  }

  public static class GameState extends DSItem.Stub {
    String sessionName;
    int step;
    @Override
    public String id() {
      return sessionName + "-" + step;
    }

    GameState(String sessionName, int step) {
      this.sessionName = sessionName;
      this.step = step;
    }
  }
}
