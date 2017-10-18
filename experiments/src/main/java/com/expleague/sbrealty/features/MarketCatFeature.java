package com.expleague.sbrealty.features;

import com.expleague.sbrealty.Deal;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.ml.data.tools.DataTools;
import com.expleague.sbrealty.FeatureBuilder;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;

import java.util.*;

import static com.spbsu.commons.math.vectors.MxTools.*;
import static java.lang.Math.log;

/**
 * Experts League
 * Created by solar on 10.06.17.
 */
public abstract class MarketCatFeature extends FeatureBuilder<Deal> implements Evaluator<Deal> {
  protected final List<DateState> dates = new ArrayList<>();
  private final long[] ts;
  private Vec solution;

  protected MarketCatFeature(String id, String description) {
    super(id, description, null);
    calc = this;
    final TDoubleArrayList monthUsdWindow = new TDoubleArrayList(31);
    final double[] usdSum2 = {0};
    final double[] usdSum = {0};
    DataTools.readCSVWithHeader("./experiments/data/sbrealty/macro.csv.gz", resolve -> {
      final DateState.Builder builder = new DateState.Builder();
      try {
        builder.date(resolve.asDate("timestamp", "yyyy-MM-dd"));
        double usdrub = resolve.asDouble("usdrub");
        builder.usd(usdrub);
        double rent_price_1room_eco = resolve.asDouble("rent_price_1room_eco");
        double mortgage = resolve.asDouble("mortgage_rate");
        if (rent_price_1room_eco == 2.31)
          rent_price_1room_eco += 30;
        builder.rent1(rent_price_1room_eco);
        monthUsdWindow.add(usdrub);
        usdSum2[0] += usdrub * usdrub;
        usdSum[0] += usdrub;
        if (monthUsdWindow.size() > 30) {
          double v = monthUsdWindow.removeAt(0);
          usdSum[0] -= v;
          usdSum2[0] -= v * v;
        }
        builder.salary(resolve.asDouble("salary"));
        builder.mortgage(mortgage);
        builder.volatility(Math.sqrt((usdSum2[0] - usdSum[0] * usdSum[0] / monthUsdWindow.size())) / monthUsdWindow.size());
        dates.add(builder.build());
      }
      catch (NumberFormatException ignore) {
//          System.out.println(ignore);
      }
    });
    ts = dates.stream().mapToLong(DateState::date).toArray();
  }

  public void init(List<Deal> deals, Vec signal, Date start, Date end) {
    final Calendar calendar = Calendar.getInstance();
    calendar.setTimeInMillis(dates.get(0).date());
    int index = 0;
    double mean = 140000;
    double tau = 1. / (50000. * 50000.);
    double tau_0 = 1 / (50000. * 50000.);
    final VecBuilder Abuilder = new VecBuilder();
    final VecBuilder bbuilder = new VecBuilder();
    while (calendar.getTime().before(end) && index < deals.size()) {
      while (index < deals.size() && calendar.getTimeInMillis() > deals.get(index).dateTs()) {
        final Deal current = deals.get(index);
        final double price = signal.get(index);
        final double pricePerSqMeter = price / current.area();
        if (Math.abs(pricePerSqMeter - mean) < 3 * Math.sqrt(1 / tau) || pricePerSqMeter < 50000) {
          mean = (tau_0 * mean + tau * pricePerSqMeter) / (tau + tau_0);
          tau_0 += tau;
          tau_0 /= 1.001 * 1.001;
        }
        index++;
      }
      calendar.add(Calendar.DAY_OF_MONTH, 1);
//      System.out.print(dateFormat.format(calendar.getTime()) + "\t" + mean + "\t" + Math.sqrt(1 / tau_0) + "\t" + windowPrice / lastPrices.size());
      if (index > 0) {
        final DateState dateState = market(calendar.getTime());
        marketFeatures(Abuilder, dateState);
        bbuilder.append(mean);
        dates.add(dateState);
      }
//      System.out.println("\t" + dateState.usd() + "\t" + dateState.volatility() + "\t" + dateState.rent1() + "\t" + dateState.salary());
    }

    final Vec b = bbuilder.build();
    final Vec build = Abuilder.build();
    final VecBasedMx A = new VecBasedMx(build.length() / b.length(), build);
    multiply(transpose(A), A);
    Mx inverseATA = inverse(multiply(transpose(A), A));
    solution = multiply(inverseATA, multiply(transpose(A), b));
    System.out.println("Residue: " + Math.sqrt(VecTools.distanceL2(b, multiply(A, solution)) / b.dim()));
  }

  private static void marketFeatures(VecBuilder builder, DateState state) {
    builder.append(state.usd());
    builder.append(log(state.volatility()));
    builder.append(state.rent1());
    builder.append(state.salary());
    builder.append(state.mortgage());
    builder.append(state.usd() * log(state.volatility()));
    builder.append(state.rent1() * log(state.volatility()));
    builder.append(state.usd() * state.rent1());
    builder.append(state.rent1() * 1000 / state.salary());
  }

  protected DateState market(Date onDate) {
    final int index = Arrays.binarySearch(ts, onDate.getTime());
    if (index >= 0)
      return dates.get(index);
    return dates.get(-index - 1);
  }

  public double squareMeterPrice(DateState ds) {
    final VecBuilder builder = new VecBuilder(10);
    marketFeatures(builder, ds);
    return VecTools.multiply(builder.build(), solution);
  }


  /**
   * Experts League
   * Created by solar on 09.06.17.
   */
  public static class DateState {
    private long date;
    private double usd;
    private double usdVolatility;
    private double rent1;
    private double salary;
    private double mortgage;
    private TIntDoubleHashMap typeMultipliers;
    private TObjectDoubleHashMap<String> districtMultipliers;

    public long date() {
      return date;
    }

    public double usd() {
      return usd;
    }

    public double volatility() {
      return usdVolatility;
    }

    public double rent1() {
      return rent1;
    }

    public double salary() {
      return salary;
    }

    public double mortgage() {
      return mortgage;
    }

    public void typeMultipliers(TIntDoubleHashMap multipliers) {
      typeMultipliers = multipliers;
    }

    public double typeMultiplier(int type) {
      if (typeMultipliers == null)
        System.out.println();
      return typeMultipliers.get(type);
    }

    public void districtMultipliers(TObjectDoubleHashMap<String> multipliers) {
      districtMultipliers = multipliers;
    }

    public double districtMultiplier(String district) {
      return districtMultipliers.get(district);
    }

    public boolean emptyTypeMultipliers() {
      return typeMultipliers == null;
    }

    public boolean emptyDistrictMultipliers() {
      return districtMultipliers == null;
    }

    public static class Builder {
      DateState result = new DateState();

      public DateState build() {
        DateState result = this.result;
        this.result = new DateState();
        return result;
      }

      public void date(Date timestamp) {
        result.date = timestamp.getTime();
      }

      public void usd(double usdrub) {
        result.usd = usdrub;
      }

      public void rent1(double rent) {
        result.rent1 = rent;
      }

      public void volatility(double v) {
        result.usdVolatility = v;
      }

      public void salary(double salary) {
        result.salary = salary;
      }

      public void mortgage(double mortgage) {
        result.mortgage = mortgage;
      }
    }
  }
}
