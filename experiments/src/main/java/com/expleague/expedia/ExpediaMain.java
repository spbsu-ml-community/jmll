package com.expleague.expedia;

import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.logging.Interval;

import java.io.IOException;

public class ExpediaMain {
  public static void main(String[] args) throws IOException {
    FastRandom rng = new FastRandom();

    Interval.start();

    switch (args[0]) {
      case "pool": {
        // args example: pool <path to train data> <path to store pool> <path to store builders>
        ExpediaPoolBuilder.build(args[1], args[2], args[3]);
        break;
      }
    }

    Interval.stopAndPrint();
  }
}
