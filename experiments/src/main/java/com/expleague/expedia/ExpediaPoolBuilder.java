package com.expleague.expedia;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.util.logging.Logger;
import com.expleague.expedia.features.CTRBuilder;
import com.expleague.ml.data.tools.CsvRow;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.data.tools.PoolBuilder;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.function.Consumer;
import java.util.zip.GZIPOutputStream;

import static com.expleague.expedia.features.CTRBuilder.load;

public class ExpediaPoolBuilder {
  private static final int DUMP = 100_000;
  private static final Logger LOG = Logger.create(ExpediaPoolBuilder.class);

  private static final String[] COLUMNS = new String[]{
          "date_time",
          "user_id",
          "user_location_city",
          "srch_ci",
          "srch_co",
          "srch_destination_id",
          "hotel_cluster",
          "is_booking"
  };

  private static final String[] META = new String[]{
          "day-ctr", "CTR built on days",
          "user-ctr", "CTR built on users",
          "user-city-ctr", "CTR built on users' cities",
          "ci-ctr", "CTR built on checkin dates",
          "co-ctr", "CTR built on checkout dates",
          "dest-ctr", "CTR built on destinations",
          "hotel-ctr", "CTR built on hotels",
          "booked", "If the user booked the hotel"
  };

  private static final int BUILDERS_COUNT = 7;

  public static Pool<EventItem> buildTrain(final String trainPath, final String builderPath) throws IOException {
    final ArrayList<CTRBuilder<Integer>> builders = new ArrayList<>();
    final VecBuilder booked = new VecBuilder();

    LOG.debug("Create features...");
    for (int i = 0; i < BUILDERS_COUNT; ++i) {
      builders.add(new CTRBuilder<>(META[2 * i], META[2 * i + 1]));
    }

    LOG.debug("Process data...");
    final PoolBuilder builder = new PoolBuilder();
    DataTools.readCSVWithHeader(trainPath, new Consumer<CsvRow>() {
      private int index = 0;
      private int[] values = new int[COLUMNS.length];

      @Override
      public void accept(final CsvRow row) {
        for (int i = 0; i < COLUMNS.length; ++i) {
          values[i] = row.asInt(COLUMNS[i]);
        }

        final int hasBooked = values[COLUMNS.length - 1];

        for (int i = 0; i < BUILDERS_COUNT; ++i) {
          builders.get(i).add(values[i], hasBooked > 0);
        }

        // add (day, user, hotel)
        builder.addItem(new EventItem(values[0], values[1], values[6]));
        booked.append(hasBooked);

        if (++index % DUMP == 0) {
          System.out.println("Processed: " + index);
        }
      }
    });

    builder.setMeta(getDataSetMeta());

    // add new features
    for (final CTRBuilder<Integer> ctr : builders) {
      builder.newFeature(ctr.getMeta(), ctr.build());
    }

    // add booked feature
    final JsonTargetMeta bookedMeta = getTargetMeta(META[2 * BUILDERS_COUNT], META[2 * BUILDERS_COUNT + 1]);
    builder.newTarget(bookedMeta, booked.build());

    // save builders
    LOG.debug("Save builders...");
    for (final CTRBuilder<Integer> ctr : builders) {
      ctr.write(builderPath);
    }

    return (Pool<EventItem>) builder.create();
  }

  public static Pool<EventItem> buildValidate(final String validatePath, final String builderPath) throws IOException, ClassNotFoundException {
    final ArrayList<CTRBuilder<Integer>> builders = new ArrayList<>();

    LOG.debug("Load builders...");
    for (int i = 0; i < BUILDERS_COUNT; ++i) {
      final CTRBuilder<Integer> builder = CTRBuilder.<Integer>load(builderPath, META[2 * i]);
      builders.add(builder);
    }

    final VecBuilder booked = new VecBuilder();

    LOG.debug("Process data...");
    final PoolBuilder builder = new PoolBuilder();
    DataTools.readCSVWithHeader(validatePath, new Consumer<CsvRow>() {
      private int index = 0;
      private int[] values = new int[COLUMNS.length];

      @Override
      public void accept(final CsvRow row) {
        for (int i = 0; i < COLUMNS.length; ++i) {
          values[i] = row.asInt(COLUMNS[i]);
        }

        for (int i = 0; i < BUILDERS_COUNT; ++i) {
          builders.get(i).addCTR(values[i]);
        }

        // add (day, user, hotel)
        builder.addItem(new EventItem(values[0], values[1], values[6]));
        booked.append(values[COLUMNS.length - 1]);

        if (++index % DUMP == 0) {
          System.out.println("Processed: " + index);
        }
      }
    });

    builder.setMeta(getDataSetMeta());

    // add new features
    for (final CTRBuilder<Integer> ctr : builders) {
      builder.newFeature(ctr.getMeta(), ctr.build());
    }

    // add booked feature
    final JsonTargetMeta bookedMeta = getTargetMeta(META[2 * BUILDERS_COUNT], META[2 * BUILDERS_COUNT + 1]);
    builder.newTarget(bookedMeta, booked.build());

    return (Pool<EventItem>) builder.create();
  }

  // TODO: replace bad stuff...
  public static void buildWithFactor(final Pool<EventItem> pool, final Vec factor, final String poolPath) throws IOException {
    final PoolBuilder builder = new PoolBuilder();

    // set new meta
    builder.setMeta(getDataSetMeta());

    // add items
    for (int i = 0; i < pool.size(); ++i) {
      builder.addItem(pool.data().at(i));
    }

    // add features
    for (int i = 0; i < pool.features().length; ++i) {
      final JsonFeatureMeta meta = (JsonFeatureMeta) pool.features()[i];
      final double[] values = pool.vecData().data().col(i).toArray();
      builder.newFeature(meta, new ArrayVec(values, 0, values.length));
    }

    // add new feature
    final JsonFeatureMeta factorMeta = getFeatureMeta("factor", "Our factor");
    builder.newFeature(factorMeta, factor);

    // add target
    final JsonTargetMeta targetMeta = getTargetMeta("booked", "If the user booked the hotel");
    builder.newTarget(targetMeta, pool.target(0));

    // save data pool
    try (final Writer out = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(poolPath)))) {
      DataTools.writePoolTo(builder.create(), out);
    }
  }

  private static JsonDataSetMeta getDataSetMeta() {
    return new JsonDataSetMeta("Expedia",
            System.getProperty("user.name"),
            new Date(),
            EventItem.class,
            "expedia-" + DateFormat.getInstance().format(new Date())
    );
  }

  private static JsonFeatureMeta getFeatureMeta(final String id, final String description) {
    final JsonFeatureMeta meta = new JsonFeatureMeta();
    meta.id = id;
    meta.description = description;
    meta.type = FeatureMeta.ValueType.VEC;
    return meta;
  }

  private static JsonTargetMeta getTargetMeta(final String id, final String description) {
    final JsonTargetMeta meta = new JsonTargetMeta();
    meta.id = id;
    meta.description = description;
    meta.type = FeatureMeta.ValueType.VEC;
    return meta;
  }
}
