package com.expleague.crawl.data.impl;

import com.expleague.crawl.data.impl.system.EmptyFieldsDefault;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.expleague.crawl.data.Message;

public class PackedCellMessage implements Message {
  private static long FOREGROUND_MM_UNSEEN  = 0x00020000;
  private static long FOREGROUND_UNSEEN  = 0x00040000;
  //TODO: wtf it is. see tileweb.cc 1131 (seems like "parent" for random items)
  @JsonProperty("base")
  private int base = EmptyFieldsDefault.emptyInt();
  @JsonProperty("fg")
  private PackedUnsignedLong foreground = EmptyFieldsDefault.emptyValue();
  @JsonProperty("bg")
  private PackedUnsignedLong background = EmptyFieldsDefault.emptyValue();

  public boolean merge(final PackedCellMessage target) {
    boolean updated = false;
    if (target.hasBase()) {
      base = target.getBase();
      updated = true;
    }
    if (target.hasBackground()) {
      background = target.background();
      updated = true;
    }

    if (target.hasForeground()) {
      foreground = target.foreground();
      updated = true;
    }

    return updated;
  }

  static boolean visible(final PackedCellMessage cell) {
//    return foreground.
    return true;

  }

  public boolean hasBase() {
    return EmptyFieldsDefault.notEmpty(base);
  }

  public int getBase() {
    return base;
  }

  public boolean hasBackground() {
    return EmptyFieldsDefault.notEmpty(background());
  }

  public PackedUnsignedLong background() {
    return background;
  }

  public boolean hasForeground() {
    return EmptyFieldsDefault.notEmpty(foreground());
  }

  public PackedUnsignedLong foreground() {
    return foreground;
  }

  //Reserve for future. We don't use all this stuff
  @JsonProperty("cloud")
  private boolean isCloud;

  @JsonProperty("bloody")
  private boolean isBloody;

  @JsonProperty("old_blood")
  private boolean hasOldBlood;

  @JsonProperty("silenced")
  private boolean isSilenced;

  @JsonProperty("halo")
  private int halo;

  @JsonProperty("moldy")
  private boolean isMoldy;

  @JsonProperty("glowing_mold")
  private boolean isGlowingMold;

  @JsonProperty("sanctuary")
  private boolean isSanctuary;

  @JsonProperty("liquefied")
  private boolean isLiquefied;

  @JsonProperty("orb_glow")
  private int orbGlow;

  @JsonProperty("quad_glow")
  private boolean hasOrbGlow;

  @JsonProperty("disjunct")
  private boolean isDisjunct;

  @JsonProperty("mangrove_water")
  private boolean isMangroveWater;

  @JsonProperty("blood_rotation")
  private int bloodRotation;

  //i think, we don't need this
  @JsonProperty("travel_trail")
  private int travelTrail;

  @JsonProperty("heat_aura")
  private int heatAura;

  @JsonProperty("flv")
  private FlavourMessage flavour;


  public boolean isCloud() {
    return isCloud;
  }

  public boolean isBloody() {
    return isBloody;
  }

  public boolean isHasOldBlood() {
    return hasOldBlood;
  }

  public boolean isSilenced() {
    return isSilenced;
  }

  public int getHalo() {
    return halo;
  }

  public boolean isMoldy() {
    return isMoldy;
  }

  public boolean isGlowingMold() {
    return isGlowingMold;
  }

  public boolean isSanctuary() {
    return isSanctuary;
  }

  public boolean isLiquefied() {
    return isLiquefied;
  }

  public int getOrbGlow() {
    return orbGlow;
  }

  public boolean isHasOrbGlow() {
    return hasOrbGlow;
  }

  public boolean isDisjunct() {
    return isDisjunct;
  }

  public boolean isMangroveWater() {
    return isMangroveWater;
  }

  public int getBloodRotation() {
    return bloodRotation;
  }

  public int getTravelTrail() {
    return travelTrail;
  }

  public int getHeatAura() {
    return heatAura;
  }

  public FlavourMessage getFlavour() {
    return flavour;
  }


  //TODO: mcache, doll if we need them. line 1200+
}
