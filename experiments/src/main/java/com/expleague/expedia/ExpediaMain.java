package com.expleague.expedia;

import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.logging.Interval;
import com.expleague.expedia.features.Factor;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.zip.GZIPInputStream;

public class ExpediaMain {
  private static final int DUMP = 10_000;

  public static void main(String[] args) throws IOException {
    Interval.start();

    switch (args[0]) {
      case "pool": {
        // args example: pool <path to train data> <path to store pool> <path to store builders>
        ExpediaPoolBuilder.build(args[1], args[2], args[3]);
        break;
      }
      case "factor": {
        // args example: factor <path to pool> <path to new pool> <path to store factor>
        try (final Reader in = new InputStreamReader(new GZIPInputStream(new FileInputStream(args[1])))) {
          final Pool<EventItem> pool = DataTools.readPoolFrom(in);
          final Seq<Double> target = (Seq<Double>) pool.target("booked");

          final Factor factor = new Factor();

          for (int eventIndex = 0; eventIndex < pool.size(); ++eventIndex) {
            final EventItem event = pool.data().at(eventIndex);
            final int hasBooked = target.at(eventIndex).intValue();
            factor.add(event.user, event.hotel, hasBooked);

            if (eventIndex % DUMP == 0) {
              System.out.println("Processed: " + eventIndex);
            }
          }

          // save factor
          factor.write(args[3]);

          ExpediaPoolBuilder.buildWithFactor(pool, factor.build(), args[2]);
        }
        break;
      }
    }

    Interval.stopAndPrint();
  }
}
