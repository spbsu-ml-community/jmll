package com.expleague.sbrealty;

import com.expleague.sbrealty.features.BuildingTypeFeature;
import com.expleague.sbrealty.features.DistrictFeature;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.ml.data.tools.CsvRow;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.PoolBuilder;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.JsonDataSetMeta;
import com.spbsu.ml.meta.impl.JsonTargetMeta;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.util.*;

import static java.lang.Math.log;

/**
 * Experts League
 * Created by solar on 05.06.17.
 */
public class SBRealty {
  public static void main(String[] args) throws IOException {
    if (args.length < 1 || "pool".equals(args[0])) {
      final Calendar calendar = Calendar.getInstance();
      calendar.set(Calendar.YEAR, 2015);
      calendar.set(Calendar.MONTH, 2);
      calendar.set(Calendar.DAY_OF_MONTH, 1);
      final Date validate = calendar.getTime();

      final List<Deal> allDeals = new ArrayList<>();
      final VecBuilder allPrice = new VecBuilder();
      final List<Deal> trash = new ArrayList<>();
      final VecBuilder learnPrice = new VecBuilder();
      final PoolBuilder learnBuilder = new PoolBuilder();
      final VecBuilder validatePrice = new VecBuilder();
      final PoolBuilder validateBuilder = new PoolBuilder();
      DataTools.readCSVWithHeader("./experiments/data/sbrealty/train.csv.gz", (CsvRow resolve) -> {
        final Deal.Builder builder = new Deal.Builder();
        double price_doc = resolve.asDouble("price_doc");

        builder.id(resolve.asString("id"));
        builder.area(resolve.asDouble("full_sq"));
        builder.livingArea(resolve.asDouble("life_sq", -1));
        builder.floor(resolve.asInt("floor", -1));
        builder.maxFloor(resolve.asInt("max_floor", -1));
        builder.roomsCount(resolve.asInt("num_room", -1));
        builder.kitchenArea(resolve.asDouble("kitch_sq", -1));
        builder.type(resolve.asInt("material", -1));
        builder.date(resolve.asDate("timestamp", "yyyy-MM-dd"));
        builder.district(resolve.asString("sub_area"));
        final Deal deal;
        if (builder.valid()) {
          deal = builder.build();
          if (deal.dateTs() >= validate.getTime()) {
            validateBuilder.addItem(deal);
            validatePrice.append(price_doc);
          }
          else {
            learnBuilder.addItem(deal);
            learnPrice.append(price_doc);
          }
        }
        else {
          deal = builder.build();
          trash.add(deal);
        }
        allDeals.add(deal);
        allPrice.add(price_doc);
      });


      allDeals.sort(Comparator.comparingLong(Deal::dateTs));

      System.out.println("Semitrash: " + trash.stream().filter(deal -> new Deal.Builder(deal).semiValid()).count());
      System.out.println("Quite totally trash: " +
          trash.stream()
              .filter(deal -> !new Deal.Builder(deal).semiValid())
              .filter(deal -> new Deal.Builder(deal).quiteNotValid())
              .count()
      );

      final Date start = new Date(learnBuilder.<Deal>items().mapToLong(Deal::dateTs).min().orElse(validate.getTime()));
      final Date end = new Date(validateBuilder.<Deal>items().mapToLong(Deal::dateTs).max().orElse(validate.getTime()));
      for (final FeatureBuilder<Deal> featureBuilder : ALL) {
        featureBuilder.init(allDeals, allPrice.build(), start, validate);
      }
      buildPool(learnPrice, learnBuilder, "learn");
      buildPool(validatePrice, validateBuilder, "validate");
    }
  }

  private static void buildPool(VecBuilder validatePrice, PoolBuilder validateBuilder, String id) throws IOException {
    // validate
    {
      final JsonDataSetMeta meta = new JsonDataSetMeta("https://www.kaggle.com/c/sberbank-russian-housing-market",
          System.getProperty("user.name"),
          new Date(),
          Deal.class,
          "sb-realty-" + id + "-" + DateFormat.getInstance().format(new Date())
      );
      validateBuilder.setMeta(meta);
    }
    validateBuilder.<Deal>items().forEach(deal -> {
      for (FeatureBuilder<Deal> builder : ALL) {
        builder.accept(deal);
      }
    });
    for (final FeatureBuilder<Deal> featureBuilder : ALL) {
      validateBuilder.newFeature(featureBuilder.meta(), featureBuilder.build());
    }
    final Vec target = validatePrice.build();
    {
      final JsonTargetMeta meta = new JsonTargetMeta();
      meta.id = "price";
      meta.description = "original price from data";
      meta.type = FeatureMeta.ValueType.VEC;
      validateBuilder.newTarget(meta, target);
    }
    {
      final VecBuilder logTargetBuilder = new VecBuilder();
      for (int i = 0; i < target.dim(); i++) {
        logTargetBuilder.append(log(target.get(i)));
      }
      final JsonTargetMeta meta = new JsonTargetMeta();
      meta.id = "log-price";
      meta.description = "log price";
      meta.type = FeatureMeta.ValueType.VEC;
      validateBuilder.newTarget(meta, logTargetBuilder.build());
    }

    DataTools.writePoolTo(validateBuilder.create(Deal.class), new FileWriter("./experiments/data/sbrealty/" + id + ".pool"));
  }

  @SuppressWarnings("unchecked")
  private static FeatureBuilder<Deal>[] ALL = new FeatureBuilder[]{
      new FeatureBuilder<>("area", "", Deal::area),
      new FeatureBuilder<>("floor", "", Deal::floor),
      new FeatureBuilder<Deal>("last-floor", "", d -> d.floor() == d.maxFloor() ? 1 : 0),
      new FeatureBuilder<>("kitchen-area", "", Deal::kitchenArea),
      new FeatureBuilder<>("rooms-count", "", Deal::roomsCount),
      new FeatureBuilder<>("living-area", "", Deal::larea),
      new BuildingTypeFeature(),
      new DistrictFeature(),
  };
}
