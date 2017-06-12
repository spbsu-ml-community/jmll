package com.spbsu.sbrealty.features;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.sbrealty.Deal;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TObjectDoubleHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.Calendar;
import java.util.Date;
import java.util.List;

/**
 * Experts League
 * Created by solar on 10.06.17.
 */
public class DistrictFeature extends MarketCatFeature {
  public DistrictFeature() {
    super("district-ctr", "");
  }

  @Override
  public void init(List<Deal> deals, Vec signal, Date start, Date end) {
    super.init(deals, signal, start, end);
    final Calendar calendar = Calendar.getInstance();
    calendar.setTime(start);
    int index = 0;
    TObjectDoubleHashMap<String> sums = new TObjectDoubleHashMap<>();
    TObjectIntHashMap<String> counts = new TObjectIntHashMap<>();
    while (calendar.getTime().before(end) && index < deals.size()) {
      final DateState market = market(calendar.getTime());
      double meanPrice = squareMeterPrice(market);
      while (index < deals.size() && calendar.getTimeInMillis() > deals.get(index).dateTs()) {
        final Deal current = deals.get(index);
        final double price = signal.get(index) / current.area();
        sums.adjustOrPutValue(current.district(), price / meanPrice, price/meanPrice);
        counts.adjustOrPutValue(current.district(), 1, 1);
        index++;
      }
      TObjectDoubleHashMap<String> multipliers = new TObjectDoubleHashMap<>();
      sums.forEachEntry((type, sum) -> {
        multipliers.put(type, (sum + 1) / (counts.get(type) + 1));
        return true;
      });
      market.districtMultipliers(multipliers);
      calendar.add(Calendar.DATE, 1);
    }
    final TObjectDoubleHashMap<String> multipliers = new TObjectDoubleHashMap<>();
    sums.forEachEntry((type, sum) -> {
      multipliers.put(type, (sum + 1) / (counts.get(type) + 1));
      return true;
    });
    dates.stream().filter(DateState::emptyDistrictMultipliers).forEach(state -> state.districtMultipliers(multipliers));

    while (calendar.getTime().before(end)) {
      final DateState market = market(calendar.getTime());
      market.districtMultipliers(multipliers);
    }
  }

  @Override
  public double value(Deal deal) {
    final DateState market = market(new Date(deal.dateTs()));
    double multiplier = market.districtMultiplier(deal.district());
    return multiplier > 0 ? multiplier : 1;
  }
}
