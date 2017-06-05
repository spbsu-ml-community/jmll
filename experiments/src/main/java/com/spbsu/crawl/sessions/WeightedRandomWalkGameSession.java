package com.spbsu.crawl.sessions;

import com.spbsu.commons.func.types.impl.TypeConvertersCollection;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import com.spbsu.crawl.bl.GameSession;
import com.spbsu.crawl.bl.Hero;
import com.spbsu.crawl.bl.Mob;
import com.spbsu.crawl.bl.crawlSystemView.MobsListener;
import com.spbsu.crawl.bl.events.HeroListener;
import com.spbsu.crawl.bl.events.MapListener;
import com.spbsu.crawl.bl.map.CrawlGameSessionMap;
import com.spbsu.crawl.bl.map.Position;
import com.spbsu.crawl.bl.map.TerrainType;
import com.spbsu.crawl.bl.GameSession;
import com.spbsu.crawl.bl.Hero;
import com.spbsu.crawl.learning.features.Feature;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.io.*;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class WeightedRandomWalkGameSession implements GameSession, MapListener, HeroListener, MobsListener {
  private static final Logger LOG = Logger.getLogger(WeightedRandomWalkGameSession.class.getName());
  private final CrawlGameSessionMap crawlMap = new CrawlGameSessionMap();
  private int x;
  private int y;
  private double increment = 0.2;
  private double prevScore = 0;
  private double step = 0.1;
  private int turn = 0;
  private int hp;

  public void alter(double score) {
    if (score > prevScore) {
      increment += step;
    } else {
      step *= -0.9;
      increment += step;
    }
    prevScore = score;
  }

  @Override
  public void observeMonster(final Mob mob) {
    LOG.log(Level.ALL, "Observe monster " + mob.toString());
  }

  @Override
  public void lostMonster(final Mob mob) {
    LOG.log(Level.ALL, "Lost monster " + mob.toString());
  }


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
    return !terrain.isPresent() || terrain.get() != TerrainType.WALL;
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
          return false;
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
  public Mob.Action action() {
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
  public Hero.Stat chooseStatForUpgrade() {
    return Hero.Stat.Strength;
  }

  @Override
  public void features(List<Feature> features) {
    final VecBuilder builder = new VecBuilder();
    for (Feature feature : features) {
      for (int i = 0; i < feature.dim(); i++)
        builder.add((double)feature.at(i));
    }
    try {
      final Vec vec = builder.build();
//      if (vec.dim() > 5)
//        System.out.println();
      featuresWriter.append(TypeConvertersCollection.ROOT.convert(vec, String.class));
      featuresWriter.append('\n');
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void finish() {
    try {
      featuresWriter.flush();
      featuresWriter.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void tile(final Position position, final TerrainType type) {
    crawlMap.tile(position.x(), position.y(), type);
  }

  @Override
  public void changeLevel(String id) {
    crawlMap.changeLevel(id);
  }

  @Override
  public void resetPosition() {
    crawlMap.resetPosition();
  }

  @Override
  public void heroPosition(int x, int y) {
    this.x = x;
    this.y = y;
  }

  @Override
  public void hp(int hp) {
    this.hp = hp;
  }

  Writer featuresWriter;
  public WeightedRandomWalkGameSession(File file) throws IOException {
    //noinspection ResultOfMethodCallIgnored
    file.createNewFile();
    featuresWriter = new OutputStreamWriter(new FileOutputStream(file));
  }
}
