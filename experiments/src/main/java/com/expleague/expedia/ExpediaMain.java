package com.expleague.expedia;

import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.logging.Interval;
import com.expleague.expedia.features.Factor;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import org.apache.commons.cli.*;

import java.io.*;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class ExpediaMain {
  private static final int DUMP = 100_000;

  private static final String DATA_OPTION = "d";
  private static final String OUTPUT_OPTION = "o";
  private static final String BUILDER_OPTION = "b";
  private static final String TRAIN_OPTION = "t";
  private static final String POOL_OPTION = "p";

  private static Options options = new Options();

  static {
    options.addOption(OptionBuilder.withLongOpt("data").withDescription("path to data.csv").hasArg().create(DATA_OPTION));
    options.addOption(OptionBuilder.withLongOpt("output").withDescription("output file name").hasArg().create(OUTPUT_OPTION));
    options.addOption(OptionBuilder.withLongOpt("builder").withDescription("path to directory with builders").hasArg().create(BUILDER_OPTION));
    options.addOption(OptionBuilder.withLongOpt("train").withDescription("create new builders").hasArg(false).create(TRAIN_OPTION));
    options.addOption(OptionBuilder.withLongOpt("pool").withDescription("path to pool").hasArg().create(POOL_OPTION));
  }

  public static void main(String[] args) throws IOException {
    Interval.start();

    final CommandLineParser parser = new GnuParser();

    try {
      final CommandLine command = parser.parse(options, args);

      if (command.getArgs().length == 0) {
        throw new RuntimeException("Please provide mode to run");
      }

      final String mode = command.getArgs()[0];

      switch (mode) {
        case "pool": {
          pool(command);
          break;
        }
        case "factor": {
          factor(command);
          break;
        }

        default:
          throw new RuntimeException("Mode " + mode + " is not recognized");
      }
    } catch (ParseException e) {
      System.err.println(e.getLocalizedMessage());

      final HelpFormatter formatter = new HelpFormatter();
      formatter.printHelp("Expedia", options);
    } catch (Exception e) {
      System.err.println(e.getLocalizedMessage());
    }

    Interval.stopAndPrint();
  }

  private static void pool(final CommandLine command) throws Exception {
    if (!command.hasOption(DATA_OPTION)) {
      throw new MissingArgumentException("Please provide 'DATA_OPTION'");
    }

    if (!command.hasOption(OUTPUT_OPTION)) {
      throw new MissingArgumentException("Please provide 'OUTPUT_OPTION'");
    }

    if (!command.hasOption(BUILDER_OPTION)) {
      throw new MissingArgumentException("Please provide 'BUILDER_OPTION'");
    }

    Pool<EventItem> pool;

    if (command.hasOption(TRAIN_OPTION)) {
      pool = ExpediaPoolBuilder.buildTrain(command.getOptionValue(DATA_OPTION), command.getOptionValue(BUILDER_OPTION));
    } else {
      pool = ExpediaPoolBuilder.buildValidate(command.getOptionValue(DATA_OPTION), command.getOptionValue(BUILDER_OPTION));
    }

    writePool(pool, command.getOptionValue(OUTPUT_OPTION));
  }

  private static void factor(final CommandLine command) throws MissingArgumentException, IOException, ClassNotFoundException {
    if (!command.hasOption(POOL_OPTION)) {
      throw new MissingArgumentException("Please provide 'POOL_OPTION'");
    }

    if (!command.hasOption(OUTPUT_OPTION)) {
      throw new MissingArgumentException("Please provide 'OUTPUT_OPTION'");
    }

    if (!command.hasOption(BUILDER_OPTION)) {
      throw new MissingArgumentException("Please provide 'BUILDER_OPTION'");
    }

    final boolean trainMode = command.hasOption(TRAIN_OPTION);

    try (final Reader in = new InputStreamReader(new GZIPInputStream(new FileInputStream(command.getOptionValue(POOL_OPTION))))) {
      final Pool<EventItem> pool = DataTools.readPoolFrom(in);
      final Factor factor;

      if (trainMode) {
        factor = new Factor();
        final Seq<Double> target = (Seq<Double>) pool.target("booked");

        for (int eventIndex = 0; eventIndex < pool.size(); ++eventIndex) {
          final EventItem event = pool.data().at(eventIndex);
          final int hasBooked = target.at(eventIndex).intValue();
          factor.add(event.user, event.hotel, hasBooked);

          if (eventIndex % DUMP == 0) {
            System.out.println("Processed: " + eventIndex);
          }
        }

        // save factor
        factor.write(command.getOptionValue(BUILDER_OPTION));
      } else {
        factor = Factor.load(command.getOptionValue(BUILDER_OPTION));

        for (int eventIndex = 0; eventIndex < pool.size(); ++eventIndex) {
          final EventItem event = pool.data().at(eventIndex);
          factor.addFactor(event.user, event.hotel, 1);

          if (eventIndex % DUMP == 0) {
            System.out.println("Processed: " + eventIndex);
          }
        }
      }

      final JsonFeatureMeta meta = getFeatureMeta("factor", "Our factor");
      final Pool<EventItem> newPool = ExpediaPoolBuilder.addFeature(pool, meta, factor.build());
      writePool(newPool, command.getOptionValue(OUTPUT_OPTION));
    }
  }

  private static void writePool(final Pool<EventItem> pool, final String poolPath) throws IOException {
    try (final Writer out = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(poolPath)))) {
      DataTools.writePoolTo(pool, out);
    }
  }

  private static JsonFeatureMeta getFeatureMeta(final String id, final String description) {
    final JsonFeatureMeta meta = new JsonFeatureMeta();
    meta.id = id;
    meta.description = description;
    meta.type = FeatureMeta.ValueType.VEC;
    return meta;
  }
}
