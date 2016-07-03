package com.spbsu.crawl.bl;

import com.spbsu.crawl.data.impl.PlayerInfoMessage;

import java.util.List;

public class CrawlPlayerView {
  private Updater updater = new Updater();

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
    }

    private void updateStatus(final PlayerInfoMessage message) {
      if (message.statuses() != null) {
        status = message.statuses();
      }
    }

    private void updateStats(final PlayerInfoMessage message) {
      if (message.healthPoints() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        currentHealthPoints = message.healthPoints();
      }
      if (message.maxHealthPoints() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        totalHealthPoints = message.maxHealthPoints();
      }
      if (message.maxManaPoints() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        totalManaPoints = message.maxManaPoints();
      }
      if (message.manaPoints() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        currentManaPoints = message.manaPoints();
      }
      if (message.dexterity() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        dexterity = message.dexterity();
      }
      if (message.maxDexterity() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        maxDexterity = message.dexterity();
      }

      if (message.strength() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        strength = message.strength();
      }
      if (message.maxStrength() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        maxStrength = message.maxStrength();
      }

      if (message.intelegence() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        intelegence = message.intelegence();
      }
      if (message.maxIntelegence() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        maxIntelegence = message.maxIntelegence();
      }

      if (message.evasion() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        evasion = message.evasion();
      }

      if (message.shieldClass() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        shieldClass = message.shieldClass();
      }

      if (message.armorClass() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        armorClass = message.armorClass();
      }
    }

    private void updatePosition(final PlayerInfoMessage message) {
      if (message.position() != null) {
        x = message.position().x();
        y = message.position().y();
      }
    }


    private void updateExperience(final PlayerInfoMessage message) {
      if (message.experienceLevel() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        experienceLevel = message.experienceLevel();
      }
      if (message.nextExpLevelProgress() != PlayerInfoMessage.STAT_EMPTY_FIELD_VALUE) {
        nextExpLevelProgress = message.nextExpLevelProgress();
      }
    }
  }
}
