package com.expleague.expedia;

import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.data.tools.CsvRow;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.PoolBuilder;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.sbrealty.Deal;

import java.io.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Consumer;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class ExpediaMain {
  public static void main(String[] args) throws IOException {
    FastRandom rng = new FastRandom();

    Interval.start();

    switch (args[0]) {
      case "pool": {
        // args example: pool <path to train data> <path to store pool> <path to store builders>

        final Set<Integer> hotels = new HashSet<>();
        final Set<Integer> users = new HashSet<>();
        final VecBuilder booked = new VecBuilder();

        final CTRBuilder<Integer> dayCTR = new CTRBuilder<>("day-ctr", "CTR built on days");
        final CTRBuilder<Integer> userCTR = new CTRBuilder<>("user-ctr", "CTR built on users");
        final CTRBuilder<Integer> cityCTR = new CTRBuilder<>("user-city-ctr", "CTR built on users' cities");
        final CTRBuilder<Integer> ciCTR = new CTRBuilder<>("ci-ctr", "CTR built on checkin dates");
        final CTRBuilder<Integer> coCTR = new CTRBuilder<>("co-ctr", "CTR built on checkout dates");
        final CTRBuilder<Integer> destCTR = new CTRBuilder<>("dest-ctr", "CTR built on destinations");
        final CTRBuilder<Integer> hotelCTR = new CTRBuilder<>("hotel-ctr", "CTR built on hotels");

        final PoolBuilder builder = new PoolBuilder();
        DataTools.readCSVWithHeader(args[1], new Consumer<CsvRow>() {
          private int index = 0;

          @Override
          public void accept(CsvRow row) {
            if (rng.nextDouble() > 0.05) {
              return;
            }

            final int day = row.asInt("date_time");
            final int user = row.asInt("user_id");
            final int city = row.asInt("user_location_city");
            final int ci = row.asInt("srch_ci");
            final int co = row.asInt("srch_co");
            final int dest = row.asInt("srch_destination_id");
            final int hotel = row.asInt("hotel_cluster");
            final int isBooking = row.asInt("is_booking");

            final boolean hasBooked = isBooking > 0;

            // TODO: do we need this?
            // final String hotelDay = hotel + (day / 12);

            users.add(user);
            hotels.add(hotel);
            booked.append(isBooking);

            builder.addItem(new EventItem(day, user, hotel));

            dayCTR.add(day, hasBooked);
            userCTR.add(user, hasBooked);
            cityCTR.add(city, hasBooked);
            ciCTR.add(ci, hasBooked);
            coCTR.add(co, hasBooked);
            destCTR.add(dest, hasBooked);
            hotelCTR.add(hotel, hasBooked);

            if (++index % 1_000_000 == 0) {
              System.out.println("Processed: " + index);
            }
          }
        });

        final JsonDataSetMeta dataSetMeta = new JsonDataSetMeta("Expedia",
                System.getProperty("user.name"),
                new Date(),
                Deal.class,
                "expedia-" + DateFormat.getInstance().format(new Date())
        );
        builder.setMeta(dataSetMeta);

        builder.newFeature(dayCTR.getMeta(), dayCTR.build());
        builder.newFeature(userCTR.getMeta(), userCTR.build());
        builder.newFeature(cityCTR.getMeta(), cityCTR.build());
        builder.newFeature(ciCTR.getMeta(), ciCTR.build());
        builder.newFeature(coCTR.getMeta(), coCTR.build());
        builder.newFeature(destCTR.getMeta(), destCTR.build());
        builder.newFeature(hotelCTR.getMeta(), hotelCTR.build());

        try (final Writer out = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(args[2])))) {
          DataTools.writePoolTo(builder.create(), out);
        }

        // write CTRBuilder to file
        try {
          dayCTR.write(args[3]);
        } catch (IOException e) {
          e.printStackTrace();
        }

        // read CTRBuilder from file
        try {
          CTRBuilder<Integer> ctrBuilder = CTRBuilder.load(args[3], "day-ctr");
        } catch (Exception e) {
          e.printStackTrace();
        }

        System.out.println("Hotels: " + hotels.size() + "\nUsers: " + users.size());
        break;
      }
    }

    Interval.stopAndPrint();
  }
}
