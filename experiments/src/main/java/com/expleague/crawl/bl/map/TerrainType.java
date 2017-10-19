package com.expleague.crawl.bl.map;


import com.expleague.commons.util.ArrayTools;
import com.expleague.crawl.bl.helpers.CodeSet;
import com.expleague.crawl.data.impl.UpdateMapCellMessage;

public enum TerrainType {
  CLOSED_DOOR(new CodeSet(1)),
  OPEN_DOOR(new CodeSet(34)),
  RUINED_DOOR(new CodeSet(2)),
  SEALED_DOOR(new CodeSet(3)),
  TREE(new CodeSet(4)),
  WALL(new CodeSet(5, 6, 7, 8, 9, 10, 11, 12, 13)),
  GRATE(new CodeSet(14)),
  OPEN_SEA(new CodeSet(15)),
  LAVA_SEA(new CodeSet(16)),
  STATUE(new CodeSet(17, 18)),
  MALIGN_GATEWAY(new CodeSet(19)),
  LAVA(new CodeSet(30)),
  DEEP_WATER(new CodeSet(31)),
  SHALLOW_WATER(new CodeSet(32)),
  FLOOR(new CodeSet(33)),
  TRAP(new CodeSet(35, 36, 37, 38, 39)),
  ENTER_SHOP(new CodeSet(40)),
  ABANDONED_SHOP(new CodeSet(41)),
  STONE_STAIR_DOWN(new CodeSet(42, 43, 44, 45)),
  STONE_STAIR_UP(new CodeSet(46, 47, 48, 49)),
  ENTRANCE(new CodeSet(ArrayTools.sequence(50, 55))),
  EXIT(new CodeSet(55, 60, 61, 62)),
  STONE_ARCH(new CodeSet(56)),
  EXIT_DUNGEON(new CodeSet(60)),
  UNKNOWN_TYPE(new CodeSet(-1));

  private final CodeSet code;

  TerrainType(CodeSet code) {
    this.code = code;
  }

  public boolean contains(int idx) {
    return code.contains(idx);
  }

  public static TerrainType fromMessage(final UpdateMapCellMessage cellMessage) {
    final int dungeonFeature = cellMessage.getDungeonFeatureType();
    for (TerrainType type : TerrainType.values()) {
      if (type.contains(dungeonFeature)) {
        return type;
      }
    }
    return TerrainType.UNKNOWN_TYPE;
  }
}

