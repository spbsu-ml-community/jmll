package com.expleague.expedia;

import com.expleague.commons.func.Computable;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.CharSeqBuilder;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.logging.Interval;
import com.expleague.expedia.features.CTRBuilder;
import com.expleague.expedia.features.Factor;
import com.expleague.expedia.utils.CheckPoint;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import org.apache.commons.cli.*;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class ExpediaMain {
  private static final int DUMP_STEP = 100_000;
  private static final int HOTELS_COUNT = 100;

  private static final String DATA_OPTION = "d";
  private static final String OUTPUT_OPTION = "o";
  private static final String BUILDER_OPTION = "b";
  private static final String TRAIN_OPTION = "t";
  private static final String TEST_OPTION = "test";
  private static final String POOL_OPTION = "p";

  private static final String MODEL_OPTION = "m";
  private static final String GRID_OPTION = "g";
  private static final String HOTELS_OPTION = "h";

  private static Options options = new Options();

  static {
    options.addOption(OptionBuilder.withLongOpt("data").withDescription("path to data.csv").hasArg().create(DATA_OPTION));
    options.addOption(OptionBuilder.withLongOpt("output").withDescription("output file name").hasArg().create(OUTPUT_OPTION));
    options.addOption(OptionBuilder.withLongOpt("builder").withDescription("path to directory with builders").hasArg().create(BUILDER_OPTION));
    options.addOption(OptionBuilder.withLongOpt("train").withDescription("create new builders").hasArg(false).create(TRAIN_OPTION));
    options.addOption(OptionBuilder.withLongOpt("test").withDescription("build test pool").hasArg(false).create(TEST_OPTION));
    options.addOption(OptionBuilder.withLongOpt("pool").withDescription("path to pool").hasArg().create(POOL_OPTION));

    options.addOption(OptionBuilder.withLongOpt("model").withDescription("path to model").hasArg().create(MODEL_OPTION));
    options.addOption(OptionBuilder.withLongOpt("grid").withDescription("path to grid").hasArg().create(GRID_OPTION));
    options.addOption(OptionBuilder.withLongOpt("hotel").withDescription("path to file with hotels").hasArg().create(HOTELS_OPTION));
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
        case "apply": {
          apply(command);
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
      e.printStackTrace();
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
      final int isTest = command.hasOption(TEST_OPTION) ? 1 : 0;
      pool = ExpediaPoolBuilder.buildValidate(command.getOptionValue(DATA_OPTION), command.getOptionValue(BUILDER_OPTION), isTest);
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

      final CheckPoint checkPoint = new CheckPoint(pool.size(), DUMP_STEP);

      if (trainMode) {
        factor = new Factor();
        final Seq<Double> target = (Seq<Double>) pool.target("booked");

        for (int eventIndex = 0; eventIndex < pool.size(); ++eventIndex) {
          final EventItem event = pool.data().at(eventIndex);
          final int hasBooked = target.at(eventIndex).intValue();
          factor.addEvent(event.user, event.hotel, hasBooked);

          checkPoint.check();
        }

        System.out.println("Processed all samples!");

        // save factor
        factor.write(command.getOptionValue(BUILDER_OPTION));
        System.out.println("Saved factor!");
      } else {
        factor = Factor.load(command.getOptionValue(BUILDER_OPTION));

        for (int eventIndex = 0; eventIndex < pool.size(); ++eventIndex) {
          final EventItem event = pool.data().at(eventIndex);
          factor.addFactor(event.user, event.hotel);

          checkPoint.check();
        }

        System.out.println("Processed all samples!");
      }

      final JsonFeatureMeta meta = getFeatureMeta("factor", "Our factor");
      final Pool<EventItem> newPool = ExpediaPoolBuilder.addFeature(pool, meta, factor.build());

      factor.stop();

      System.out.println("Save new pool...");
      writePool(newPool, command.getOptionValue(OUTPUT_OPTION));
    }
  }

  private static void apply(final CommandLine command) throws MissingArgumentException, IOException, ClassNotFoundException {
    if (!command.hasOption(MODEL_OPTION)) {
      throw new MissingArgumentException("Please provide 'MODEL_OPTION'");
    }

    if (!command.hasOption(GRID_OPTION)) {
      throw new MissingArgumentException("Please provide 'GRID_OPTION'");
    }

    if (!command.hasOption(POOL_OPTION)) {
      throw new MissingArgumentException("Please provide 'POOL_OPTION'");
    }

    if (!command.hasOption(BUILDER_OPTION)) {
      throw new MissingArgumentException("Please provide 'BUILDER_OPTION'");
    }

    if (!command.hasOption(OUTPUT_OPTION)) {
      throw new MissingArgumentException("Please provide 'OUTPUT_OPTION'");
    }

    if (!command.hasOption(HOTELS_OPTION)) {
      throw new MissingArgumentException("Please provide 'HOTELS_OPTION'");
    }

    final Reader in = new InputStreamReader(new GZIPInputStream(new FileInputStream(command.getOptionValue(POOL_OPTION))));
    final Writer out = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(command.getOptionValue(OUTPUT_OPTION))));

    final Pool<EventItem> pool = DataTools.readPoolFrom(in);
    final Computable model = DataTools.readModel(new FileInputStream(command.getOptionValue(MODEL_OPTION)), new FileInputStream(command.getOptionValue(GRID_OPTION)));
    final Factor factor = Factor.load(command.getOptionValue(BUILDER_OPTION));
    final CTRBuilder<Integer> hotelCTR = CTRBuilder.load(command.getOptionValue(BUILDER_OPTION), "hotel-ctr");

    // read hotels
    final Map<Integer, int[]> hotels = readHotels(command.getOptionValue(HOTELS_OPTION));

    // the most popular hotels
    final int[] DEFAULT_HOTELS = new int[]{91, 41, 48, 64, 65};

    final double[] value = new double[HOTELS_COUNT];
    final int[] index = new int[HOTELS_COUNT];
    CharSeqBuilder output = new CharSeqBuilder();

    final CheckPoint checkPoint = new CheckPoint(pool.size(), DUMP_STEP);

    for (int eventIndex = 0; eventIndex < pool.size(); ++eventIndex) {
      final EventItem event = pool.data().at(eventIndex);
      final Vec features = pool.vecData().at(eventIndex);

      final Vec current = new ArrayVec(features.dim() + 2);

      for (int i = 0; i < features.dim(); ++i) {
        current.set(i, features.get(i));
      }

      int[] srchHotels = hotels.getOrDefault(event.hotel, DEFAULT_HOTELS);

      for (int i = 0; i < srchHotels.length; ++i) {
        final int hotel = srchHotels[i];
        current.set(features.dim(), hotelCTR.getCTR(hotel));
        current.set(features.dim() + 1, factor.getFactor(event.user, hotel));
        value[i] = -((Ensemble) model).compute(current).get(0);
        index[i] = i;
      }

      ArrayTools.parallelSort(value, index, 0, srchHotels.length);

      output.append(eventIndex).append(",");
      for (int i = 0; i < Math.min(5, srchHotels.length); ++i) {
        output.append(" ").append(srchHotels[index[i]]);
      }
      output.append("\n");

      if (checkPoint.check()) {
        out.append(output);
        output.clear();
      }
    }

    out.close();

    factor.stop();
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

  private static Map<Integer, int[]> readHotels(final String file) throws IOException {
    final Map<Integer, int[]> data = new HashMap<>();
    final Scanner scanner = new Scanner(new File(file));

    while (scanner.hasNextInt()) {
      final int srchDestinationId = scanner.nextInt();
      final int hotelsCount = scanner.nextInt();
      final int[] hotels = new int[hotelsCount];

      for (int i = 0; i < hotelsCount; ++i) {
        hotels[i] = scanner.nextInt();
      }

      data.put(srchDestinationId, hotels);
    }

    return data;
  }
}
