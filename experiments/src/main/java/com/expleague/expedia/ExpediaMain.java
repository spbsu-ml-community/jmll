package com.expleague.expedia;

import com.expleague.sbrealty.Deal;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.PoolBuilder;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;
import gnu.trove.map.hash.TObjectIntHashMap;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import java.io.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.GZIPOutputStream;

public class ExpediaMain {
  public static void main(String[] args) throws IOException {
    FastRandom rng = new FastRandom();
    Interval.start();
    switch (args[0]) {
      case "pool": {
        final Set<String> hotels = new HashSet<>();
        final Set<String> users = new HashSet<>();
        final VecBuilder booked = new VecBuilder();
        final VecBuilder hotelDateCTR = new VecBuilder();
        final VecBuilder hotelCTR = new VecBuilder();
        final VecBuilder userCTR = new VecBuilder();
        final VecBuilder countryCTR = new VecBuilder();
        final VecBuilder dayOfYear = new VecBuilder();
        final VecBuilder destinationCTR = new VecBuilder();
        final TObjectIntHashMap<String> hotelsBookingsAlpha = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> hotelsMonthBookingsAlpha = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> usersBookingsAlpha = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> locationBookingsAlpha = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> destinationBookingsAlpha = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> hotelsBookingsBetta = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> hotelsMonthBookingsBetta = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> usersBookingsBetta = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> locationBookingsBetta = new TObjectIntHashMap<>();
        final TObjectIntHashMap<String> destinationBookingsBetta = new TObjectIntHashMap<>();

        final PoolBuilder builder = new PoolBuilder();
        DataTools.readCSVWithHeader(args[1], row -> {
          final int day = row.asInt("date_time");
          final String user = row.asString("user_id");
          users.add(user);
          final String hotel = row.asString("hotel_cluster");
          hotels.add(hotel);
          final int isBooking = row.asInt("is_booking");
          final String country = row.asString("user_location_country");
          final String destination = row.asString("srch_destination_id");
          final boolean hasBooked = isBooking > 0;
          final String hotelDay = hotel + (day / 12);
          if (hasBooked) {
            hotelsBookingsAlpha.adjustOrPutValue(hotel, 1, 1);
            hotelsMonthBookingsAlpha.adjustOrPutValue(hotelDay, 1, 1);
            usersBookingsAlpha.adjustOrPutValue(user, 1, 1);
            locationBookingsAlpha.adjustOrPutValue(country, 1, 1);
            destinationBookingsAlpha.adjustOrPutValue(destination, 1, 1);
          }
          else {
            hotelsBookingsBetta.adjustOrPutValue(hotel, 1, 1);
            hotelsMonthBookingsBetta.adjustOrPutValue(hotelDay, 1, 1);
            usersBookingsBetta.adjustOrPutValue(user, 1, 1);
            locationBookingsBetta.adjustOrPutValue(country, 1, 1);
            destinationBookingsBetta.adjustOrPutValue(destination, 1, 1);
          }
          if (rng.nextDouble() < 0.1) {
            builder.addItem(new HotelUserItem(user, hotel, day));
            hotelCTR.append((hotelsBookingsAlpha.get(hotel) + 1.) / (hotelsBookingsAlpha.get(hotel) + hotelsBookingsBetta.get(hotel) + 2.));
            hotelDateCTR.append((hotelsMonthBookingsAlpha.get(hotelDay) + 1.) / (hotelsMonthBookingsAlpha.get(hotelDay) + hotelsMonthBookingsBetta.get(hotelDay) + 2.));
            userCTR.append((usersBookingsAlpha.get(hotel) + 1.) / (usersBookingsAlpha.get(hotel) + usersBookingsBetta.get(hotel) + 2.));
            countryCTR.append((locationBookingsAlpha.get(country) + 1.) / (locationBookingsAlpha.get(country) + locationBookingsBetta.get(country) + 2.));
            destinationCTR.append((destinationBookingsAlpha.get(destination) + 1.) / (destinationBookingsAlpha.get(destination) + destinationBookingsBetta.get(destination) + 2.));
            dayOfYear.append(day);
            booked.append(isBooking);
          }
        });

        final JsonDataSetMeta dsMeta = new JsonDataSetMeta("Expedia",
            System.getProperty("user.name"),
            new Date(),
            Deal.class,
            "expedia-" + DateFormat.getInstance().format(new Date())
        );
        builder.setMeta(dsMeta);
        {
          final JsonFeatureMeta meta = new JsonFeatureMeta();
          meta.id = "hotel-ctr";
          meta.description = "CTR built on hotels";
          meta.type = FeatureMeta.ValueType.VEC;
          builder.newFeature(meta, hotelCTR.build());
        }

        builder.setMeta(dsMeta);
        {
          final JsonFeatureMeta meta = new JsonFeatureMeta();
          meta.id = "hotel-month-ctr";
          meta.description = "CTR built on hotel/month pair";
          meta.type = FeatureMeta.ValueType.VEC;
          builder.newFeature(meta, hotelDateCTR.build());
        }

        {
          final JsonFeatureMeta meta = new JsonFeatureMeta();
          meta.id = "user-ctr";
          meta.description = "CTR built on users";
          meta.type = FeatureMeta.ValueType.VEC;
          builder.newFeature(meta, userCTR.build());
        }

        {
          final JsonFeatureMeta meta = new JsonFeatureMeta();
          meta.id = "day";
          meta.description = "Day of year";
          meta.type = FeatureMeta.ValueType.VEC;
          builder.newFeature(meta, dayOfYear.build());
        }

        {
          final JsonFeatureMeta meta = new JsonFeatureMeta();
          meta.id = "country";
          meta.description = "Country CTR";
          meta.type = FeatureMeta.ValueType.VEC;
          builder.newFeature(meta, countryCTR.build());
        }

        {
          final JsonFeatureMeta meta = new JsonFeatureMeta();
          meta.id = "destination";
          meta.description = "Destination CTR";
          meta.type = FeatureMeta.ValueType.VEC;
          builder.newFeature(meta, destinationCTR.build());
        }

        {
          final JsonTargetMeta meta = new JsonTargetMeta();
          meta.id = "booked";
          meta.description = "If the user booked the hotel";
          meta.type = FeatureMeta.ValueType.VEC;
          builder.newTarget(meta, booked.build());
        }

        try (final Writer out = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(args[2])))) {
          DataTools.writePoolTo(builder.create(), out);
        }

        System.out.println("Hotels: " + hotels.size() + " users: " + users.size());
        break;
      }
    }
    Interval.stopAndPrint();
  }

  @XmlRootElement
  public static class HotelUserItem extends DSItem.Stub {
    @XmlAttribute
    private String user;
    @XmlAttribute
    private String hotel;
    @XmlAttribute
    private int day;

    HotelUserItem(String user, String hotel, int day) {
      this.user = user;
      this.hotel = hotel;
      this.day = day;
    }

    HotelUserItem() {
    }

    @Override
    public String id() {
      return user + "@" + hotel + ":";
    }
  }
}
