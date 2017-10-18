package com.expleague.crawl.bl.crawlSystemView;

import com.expleague.crawl.bl.Mob;
import com.expleague.crawl.bl.map.Position;
import com.expleague.crawl.bl.map.PositionManager;
import com.expleague.crawl.data.impl.MonsterInfoMessage;
import com.expleague.crawl.data.impl.UpdateMapCellMessage;
import com.expleague.crawl.data.impl.system.EmptyFieldsDefault;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.hash.TIntHashSet;
import org.jetbrains.annotations.NotNull;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;


public class MobsView extends Subscribable.Stub<MobsListener> {
  private final TIntObjectHashMap<CrawlMob> mobs = new TIntObjectHashMap<>();
  //this damn game could use diff even for different monster types
  private final Map<Position, MonsterInfoMessage> lastMonsterStates = new HashMap<>();
  private final Updater updater = new Updater();
  private final PositionManager positionManager;

  public MobsView(final PositionManager positionManager) {
    this.positionManager = positionManager;
  }

  public Updater updater() {
    return updater;
  }

  class Updater {

    private void merge(final MonsterInfoMessage message, final MonsterInfoMessage previous) {
      if (EmptyFieldsDefault.isEmpty(message.attitude())) {
        message.setAttitude(previous.attitude());
      }
      if (EmptyFieldsDefault.isEmpty(message.baseType())) {
        message.setBaseType(previous.baseType());
      }
      if (EmptyFieldsDefault.isEmpty(message.id())) {
        message.setId(previous.id());
      }
      if (EmptyFieldsDefault.isEmpty(message.monsterStats())) {
        message.setMonsterStats(previous.monsterStats());
      }
      if (EmptyFieldsDefault.isEmpty(message.monsterType())) {
        message.setType(previous.monsterType());
      }
      if (EmptyFieldsDefault.isEmpty(message.name())) {
        message.setName(previous.name());
      }
      if (EmptyFieldsDefault.isEmpty(message.plural())) {
        message.setPlural(previous.plural());
      }
      if (EmptyFieldsDefault.isEmpty(message.threatLevel())) {
        message.setThreatLevel(previous.threatLevel());
      }

    }

    private void updateCellView(@NotNull final Position position,
                                @NotNull final MonsterInfoMessage message,
                                @NotNull final Map<Position, MonsterInfoMessage> previousMonsterStates) {
      final MonsterInfoMessage previousInfo;

      if (mobs.containsKey(message.id())) {
        final Position previousPosition = mobs.get(message.id()).position();
        previousInfo = previousMonsterStates.get(previousPosition);
      } else {
        previousInfo = previousMonsterStates.getOrDefault(position, null);
      }

      if (previousInfo != null) {
        merge(message, previousInfo);
      }
      lastMonsterStates.put(position, message);
    }


    private void updateCellsView(final List<UpdateMapCellMessage> cellMessages) {
      //we really need copy of previous map.
      // Even more - we need immutable monsterInfoMessage for previous state, so we merge to incoming message, instead of message in map
      final Map<Position, MonsterInfoMessage> previous = new HashMap<>(lastMonsterStates);

      for (final UpdateMapCellMessage cellMessage : cellMessages) {
        final Position position = positionManager.getOrCreate(cellMessage.x(), cellMessage.y());
        if (cellMessage.getMonsterInfoMessage() == null) {
          lastMonsterStates.remove(position);
        } else if (!cellMessage.getMonsterInfoMessage().isEmpty()) {
          updateCellView(position, cellMessage.getMonsterInfoMessage(), previous);
        }
      }
    }

    private void updateMobs() {
      final TIntHashSet lostMobIds = new TIntHashSet();
      for (int id : mobs.keys()) {
        lostMobIds.add(id);
      }

      for (final Map.Entry<Position, MonsterInfoMessage> entry : lastMonsterStates.entrySet()) {
        final Position position = entry.getKey();
        final MonsterInfoMessage monsterInfoMessage = entry.getValue();

        if (mobs.containsKey(monsterInfoMessage.id())) {
          final CrawlMob mob = mobs.get(monsterInfoMessage.id());
          lostMobIds.remove(monsterInfoMessage.id());
          if (position != mob.position) {
            mob.move(position);
          }
        } else {
          final CrawlMob mob = createMob(position, monsterInfoMessage);
          mobs.put(monsterInfoMessage.id(), mob);
          listeners().forEach(mobListener -> mobListener.observeMonster(mob));
        }
      }
      //if we lost mob, we'll never see him again with the same id
      lostMobIds.forEach(id -> {
        final Mob lostMob = mobs.get(id);
        mobs.remove(id);
        listeners().forEach(mobListener -> mobListener.lostMonster(lostMob));
        return true;
      });
    }

    private CrawlMob createMob(final Position position,
                               final MonsterInfoMessage monsterInfo) {
      return new CrawlMob(position,
              monsterInfo.name(),
              monsterInfo.monsterType(),
              monsterInfo.monsterStats().averageHealthPoints(),
              monsterInfo.threatLevel());
    }

    void update(final List<UpdateMapCellMessage> cellMessages) {
      updateCellsView(cellMessages);
      updateMobs();
    }

    void clear() {
      mobs.clear();
      lastMonsterStates.clear();
    }
  }

  public interface CrawlMobListener {

    void movedTo(final Position position);

  }

  static class CrawlMob extends Subscribable.Stub<CrawlMobListener> implements Mob {
    private Position position;
    private final String name;
    private final int monsterType;
    private final int averageHealthPoints;
    private final int dangerLevel;

    public CrawlMob(final Position position,
                    final String name,
                    final int monsterType,
                    final int hp,
                    final int danger) {
      this.position = position;
      this.name = name;
      this.monsterType = monsterType;
      this.averageHealthPoints = hp;
      this.dangerLevel = danger;
    }

    @Override
    public Position position() {
      return position;
    }

    @Override
    public Stream<Action> actions() {
      return Stream.empty();
    }

    private void move(final Position newPosition) {
      position = newPosition;
      listeners().forEach(listener -> listener.movedTo(newPosition));
    }


    public int monsterType() {
      return monsterType;
    }

    public String name() {
      return name;
    }

    public int averageHealthPoints() {
      return averageHealthPoints;
    }

    public int dangerLevel() {
      return dangerLevel;
    }

    @Override
    public String toString() {
      return name;
    }
  }
}
