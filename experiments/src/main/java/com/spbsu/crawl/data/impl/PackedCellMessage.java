package com.spbsu.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.spbsu.crawl.data.Message;

public class PackedCellMessage implements Message {
  //TODO: wtf it is. see tileweb.cc 1131
  @JsonProperty("base")
  private int base;

  @JsonProperty("bg")
  private long background;

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

  public int getBase() {
    return base;
  }

  public long getBackground() {
    return background;
  }

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
