package com.expleague.sbrealty.features;

import com.spbsu.commons.math.vectors.Vec;
import com.expleague.sbrealty.Deal;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.Calendar;
import java.util.Date;
import java.util.List;

/**
 * Experts League
 * Created by solar on 10.06.17.
 */
public class BuildingTypeFeature extends MarketCatFeature {
  public BuildingTypeFeature() {
    super("building-type", "");
  }

  @Override
  public void init(List<Deal> deals, Vec signal, Date start, Date end) {
    super.init(deals, signal, start, end);
    final Calendar calendar = Calendar.getInstance();
    calendar.setTime(start);
    int index = 0;
    TIntDoubleHashMap sums = new TIntDoubleHashMap();
    TIntIntHashMap counts = new TIntIntHashMap();
    while (calendar.getTime().before(end) && index < deals.size()) {
      final DateState market = market(calendar.getTime());
      double meanPrice = squareMeterPrice(market);
      while (index < deals.size() && calendar.getTimeInMillis() > deals.get(index).dateTs()) {
        final Deal current = deals.get(index);
        final double price = signal.get(index) / current.area();
        sums.adjustOrPutValue(current.type(), price / meanPrice, price/meanPrice);
        counts.adjustOrPutValue(current.type(), 1, 1);
        index++;
      }
      TIntDoubleHashMap multipliers = new TIntDoubleHashMap();
      sums.forEachEntry((type, sum) -> {
        multipliers.put(type, (sum + 1)/ (counts.get(type) + 1));
        return true;
      });
      market.typeMultipliers(multipliers);
      calendar.add(Calendar.DATE, 1);
    }
    final TIntDoubleHashMap multipliers = new TIntDoubleHashMap();
    sums.forEachEntry((type, sum) -> {
      multipliers.put(type, (sum + 1)/ (counts.get(type) + 1));
      return true;
    });
    dates.stream().filter(DateState::emptyTypeMultipliers).forEach(state -> state.typeMultipliers(multipliers));
  }

  @Override
  public double value(Deal deal) {
    final DateState market = market(new Date(deal.dateTs()));
    double v = market.typeMultiplier(deal.type());
    return v > 0 ? v : 1;
  }
}
