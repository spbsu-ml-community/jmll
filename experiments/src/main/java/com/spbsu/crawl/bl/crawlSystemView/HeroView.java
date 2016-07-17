package com.spbsu.crawl.bl.crawlSystemView;

import com.spbsu.crawl.bl.events.HeroListener;
import com.spbsu.crawl.data.impl.PlayerInfoMessage;
import com.spbsu.crawl.data.impl.system.EmptyFieldsDefault;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class HeroView extends Subscribable.Stub<HeroListener> implements Subscribable<HeroListener> {
  private Updater updater = new Updater();
  private int time = 0;
  private int turn = 0;

  private int currentHealthPoints;
  private int totalHealthPoints;

  private int currentManaPoints;
  private int totalManaPoints;

  private int gold;

  private int x;
  private int y;

  private int strength;
  private int maxStrength;

  private int intelegence;
  private int maxIntelegence;

  private int dexterity;
  private int maxDexterity;

  private int armorClass;
  private int shieldClass;
  private int evasion;

  private int experienceLevel;
  private int nextExpLevelProgress;

  private String level;
  private Set<String> statusSet = new HashSet<>();
  private Set<String> statusSetLight = new HashSet<>();

  private List<PlayerInfoMessage.PlayerStatus> status;

  public int healthPoints() {
    return currentHealthPoints;
  }

  public int manaPoints() {
    return currentManaPoints;
  }

  public int gold() {
    return gold;
  }

  public String level() {
    return level;
  }

  public int y() {
    return y;
  }

  public int x() {
    return x;
  }

  public Updater updater() {
    return updater;
  }

  public class Updater {
    public void message(final PlayerInfoMessage message) {
      updateStats(message);
      updatePosition(message);
      updateExperience(message);
      updateStatus(message);

      if (EmptyFieldsDefault.notEmpty(message.turn())) {
        time = message.time();
        turn = message.turn();
      }
    }

    private void updateStatus(final PlayerInfoMessage message) {
      if (message.statuses() != null) {
        status = message.statuses();
        for (PlayerInfoMessage.PlayerStatus st : status) {
          statusSet.add(st.text());
          statusSetLight.add(st.lightText());
        }
      }
    }

    private void updateStats(final PlayerInfoMessage message) {
      if (EmptyFieldsDefault.notEmpty(message.healthPoints())) {
        currentHealthPoints = message.healthPoints();
        listeners().forEach(heroListener -> heroListener.hp(currentHealthPoints));
      }
      if (EmptyFieldsDefault.notEmpty(message.maxHealthPoints())) {
        totalHealthPoints = message.maxHealthPoints();
      }
      if (EmptyFieldsDefault.notEmpty(message.maxManaPoints())) {
        totalManaPoints = message.maxManaPoints();
      }
      if (EmptyFieldsDefault.notEmpty(message.manaPoints())) {
        currentManaPoints = message.manaPoints();
      }
      if (EmptyFieldsDefault.notEmpty(message.dexterity())) {
        dexterity = message.dexterity();
      }
      if (EmptyFieldsDefault.notEmpty(message.maxDexterity())) {
        maxDexterity = message.dexterity();
      }
      if (EmptyFieldsDefault.notEmpty(message.strength())) {
        strength = message.strength();
      }
      if (EmptyFieldsDefault.notEmpty(message.maxStrength())) {
        maxStrength = message.maxStrength();
      }
      if (EmptyFieldsDefault.notEmpty(message.intelligence())) {
        intelegence = message.intelligence();
      }
      if (EmptyFieldsDefault.notEmpty(message.maxIntelligence())) {
        maxIntelegence = message.maxIntelligence();
      }
      if (EmptyFieldsDefault.notEmpty(message.evasion())) {
        evasion = message.evasion();
      }
      if (EmptyFieldsDefault.notEmpty(message.shieldClass())) {
        shieldClass = message.shieldClass();
      }
      if (EmptyFieldsDefault.notEmpty(message.armorClass())) {
        armorClass = message.armorClass();
      }
    }

    private void updatePosition(final PlayerInfoMessage message) {
      if (EmptyFieldsDefault.notEmpty(message.position())) {
        x = message.position().x();
        y = message.position().y();
        listeners().forEach(heroListener -> heroListener.heroPosition(x, y));
      }
    }

    private void updateExperience(final PlayerInfoMessage message) {
      if (EmptyFieldsDefault.notEmpty(message.experienceLevel())) {
        experienceLevel = message.experienceLevel();
      }
      if (EmptyFieldsDefault.isEmpty(message.nextExpLevelProgress())) {
        nextExpLevelProgress = message.nextExpLevelProgress();
      }
    }
  }
}
