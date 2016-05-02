package com.spbsu.crawl.bl.map;


import com.spbsu.commons.util.ArrayTools;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;

public enum TerrainType {
  CLOSED_DOOR(new CodeRange(1)),
  OPEN_DOOR(new CodeRange(34)),
  RUINED_DOOR(new CodeRange(2)),
  SEALED_DOOR(new CodeRange(3)),
  TREE(new CodeRange(4)),
  WALL(new CodeRange(5, 6, 7, 8, 9, 10, 11, 12, 13)),
  GRATE(new CodeRange(14)),
  OPEN_SEA(new CodeRange(15)),
  LAVA_SEA(new CodeRange(16)),
  STATUE(new CodeRange(17, 18)),
  MALIGN_GATEWAY(new CodeRange(19)),
  LAVA(new CodeRange(30)),
  DEEP_WATER(new CodeRange(31)),
  SHALLOW_WATER(new CodeRange(32)),
  FLOOR(new CodeRange(33)),
  TRAP(new CodeRange(35, 36, 37, 38, 39)),
  ENTER_SHOP(new CodeRange(40)),
  ABANDONED_SHOP(new CodeRange(41)),
  STONE_STAIR_DOWN(new CodeRange(42, 43, 44, 45)),
  STONE_STAIR_UP(new CodeRange(46, 47, 48, 49)),
  ENTRANCE(new CodeRange(ArrayTools.sequence(50, 55))),
  EXIT(new CodeRange(55, 60, 61, 62)),
  STONE_ARCH(new CodeRange(56)),
  EXIT_DUNGEON(new CodeRange(60)),
  UNKNOWN_TYPE(new CodeRange(-1));

  private final CodeRange code;

  TerrainType(CodeRange code) {
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

