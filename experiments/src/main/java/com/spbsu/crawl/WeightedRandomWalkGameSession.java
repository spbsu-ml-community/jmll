package com.spbsu.crawl;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.crawl.bl.Mob;
import com.spbsu.crawl.bl.map.CrawlGameSessionMap;
import com.spbsu.crawl.bl.map.TerrainType;
import com.spbsu.crawl.bl.map.mapEvents.MapEvent;
import com.spbsu.crawl.data.GameSession;
import com.spbsu.crawl.data.Hero;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class WeightedRandomWalkGameSession implements GameSession {
  private final CrawlGameSessionMap crawlMap = new CrawlGameSessionMap();
  private int x;
  private int y;
  private double increment = 0.2;

  class CellStats {
    int x;
    int y;
    double weight;

    public CellStats(int x, int y, double weight) {
      this.x = x;
      this.y = y;
      this.weight = weight;
    }

  }

  private TIntObjectHashMap<CellStats> stats = new TIntObjectHashMap<>();


  private int idx(int x, int y) {
    return x * 1000000 + y;
  }

  private CellStats getStats(int x, int y) {
    if (!stats.containsKey(idx(x, y))) {
      stats.put(idx(x, y), new CellStats(x, y, increment));
    }
    return stats.get(idx(x, y));
  }

  private void incWeight(int x, int y) {
    final int key = idx(x, y);
    if (stats.containsKey(key)) {
      stats.get(key).weight += increment;
    } else {
      stats.put(key, new CellStats(x, y, 2 * increment));
    }
  }


  private boolean canMoveTo(final int x, final int y) {
    Optional<TerrainType> terrain = crawlMap.terrainOnCurrentLevel(x, y);
    if (terrain.isPresent()) {
      return terrain.get() != TerrainType.WALL;
    }
    return true;
  }

  private Function<Mob.Action, Pair<Mob.Action, CellStats>> dirWeights = new Function<Mob.Action, Pair<Mob.Action, CellStats>>() {
    @Override
    public Pair<Mob.Action, CellStats> apply(Mob.Action action) {
      switch (action) {
        case MOVE_DOWN_LEFT:
          return new Pair<>(action, getStats(x - 1, y + 1));
        case MOVE_DOWN_RIGHT:
          return new Pair<>(action, getStats(x + 1, y + 1));
        case MOVE_DOWN:
          return new Pair<>(action, getStats(x, y + 1));
        case MOVE_LEFT:
          return new Pair<>(action, getStats(x - 1, y));
        case MOVE_RIGHT:
          return new Pair<>(action, getStats(x + 1, y));
        case MOVE_UP:
          return new Pair<>(action, getStats(x, y - 1));
        case MOVE_UP_LEFT:
          return new Pair<>(action, getStats(x - 1, y - 1));
        case MOVE_UP_RIGHT:
          return new Pair<>(action, getStats(x + 1, y - 1));
      }
      return null;
    }
  };


  private Predicate<Mob.Action> moveableDirection = new Predicate<Mob.Action>() {
    @Override
    public boolean test(Mob.Action action) {
      switch (action) {
        case MOVE_DOWN_LEFT:
          return canMoveTo(x - 1, y + 1);
        case MOVE_DOWN_RIGHT:
          return canMoveTo(x + 1, y + 1);
        case MOVE_DOWN:
          return canMoveTo(x, y + 1);
        case MOVE_LEFT:
          return canMoveTo(x - 1, y);
        case MOVE_RIGHT:
          return canMoveTo(x + 1, y);
        case MOVE_UP:
          return canMoveTo(x, y - 1);
        case MOVE_UP_LEFT:
          return canMoveTo(x - 1, y - 1);
        case MOVE_UP_RIGHT:
          return canMoveTo(x + 1, y - 1);
        default:
          return true;
      }
    }
  };

  @Override
  public Hero.Race race() {
    return Hero.Race.Minotaur;
  }

  @Override
  public Hero.Spec spec() {
    return Hero.Spec.Fighter_Axe;
  }


  final Mob.Action[] moves;

  {
    moves = Stream.of(Mob.Action.values()).filter(action -> action.name().startsWith("MOVE"))
            .collect(Collectors.toList()).toArray(new Mob.Action[]{});
  }

  final FastRandom rng = new FastRandom();

  @Override
  public Mob.Action tick() {
    List<Pair<Mob.Action, CellStats>> avaiableActions = Stream.of(moves).filter(moveableDirection)
            .map(dirWeights)
            .collect(Collectors.toList());
    double totalSum = 0;
    for (int i = 0; i < avaiableActions.size(); ++i) {
      totalSum += Math.exp(-(avaiableActions.get(i).getSecond().weight));
    }
    double takenWeight = totalSum * rng.nextDouble();
    int i = 0;
    while (takenWeight > 0) {
      takenWeight -= Math.exp(-avaiableActions.get(i).getSecond().weight);
      ++i;
    }
    --i;
    avaiableActions.get(i).getSecond().weight++;
    return avaiableActions.get(i).getFirst();
  }

  @Override
  public void updateMap(MapEvent event) {
    crawlMap.systemMapEvent(event);
  }

  @Override
  public void heroPosition(int x, int y) {
    this.x = x;
    this.y = y;
  }

}
