package com.spbsu.crawl.data.impl.cellCreator;

import com.spbsu.crawl.bl.map.Cell;
import com.spbsu.crawl.bl.map.Floor;
import com.spbsu.crawl.bl.map.Stair;
import com.spbsu.crawl.bl.map.Wall;
import com.spbsu.crawl.data.impl.UpdateMapCellMessage;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

/**
 * User: Noxoomo
 * Date: 01.05.16
 * Time: 0:28
 */
public class BinaryCodeToCellBuilder {
  private static TIntSet floorSet = new TIntHashSet();
  private static TIntSet upstairsSet = new TIntHashSet();
  private static TIntSet downStairsSet = new TIntHashSet();
  private static TIntSet trapsSet = new TIntHashSet();
  private static TIntSet shopsSet = new TIntHashSet();
  private static TIntSet doorsSet = new TIntHashSet();
  private static TIntSet enter = new TIntHashSet();
  private static TIntSet exit = new TIntHashSet();
  private static TIntSet altars = new TIntHashSet();
  private static TIntHashSet wallSet = new TIntHashSet();

  static {
    doorsSet.add(1);
    doorsSet.add(2);
    doorsSet.add(3);

    for (int i = 4; i < 17; ++i) {
      wallSet.add(i);
    }

    floorSet.add(31); //shallow water
    floorSet.add(33);
    floorSet.add(60); //dungeon exit
    doorsSet.add(34);

    for (int i = 35; i < 40; ++i) {
      trapsSet.add(i);
    }

    trapsSet.add(163);
    trapsSet.add(164);

    shopsSet.add(40);
    shopsSet.add(41);


    for (int i = 42; i < 46; ++i) {
      downStairsSet.add(i);
    }

    for (int i = 46; i < 50; ++i) {
      upstairsSet.add(i);
    }

    for (int i = 69; i < 86; ++i) {
      enter.add(i);
    }

    for (int i = 139; i < 149; ++i) {
      enter.add(i);
    }


    for (int i = 149; i < 159; ++i) {
      exit.add(i);
    }

    for (int i = 86; i < 102; ++i) {
      exit.add(i);
    }

    for (int i = 103; i < 121; ++i) {
      altars.add(i);
    }
    altars.add(130);
    altars.add(160);
    altars.add(161);
    altars.add(162);


  }


  public static Cell buildFromMessage(final UpdateMapCellMessage cellMessage) {
    final int dungeonFeature = cellMessage.getDungeonFeatureType();
    if (floorSet.contains(dungeonFeature) || doorsSet.contains(dungeonFeature) || altars.contains(dungeonFeature)) {
      return new Floor(cellMessage.getX(), cellMessage.getY());
    } else if (wallSet.contains(dungeonFeature)) {
      return new Wall(cellMessage.getX(), cellMessage.getY());
    } else if (upstairsSet.contains(dungeonFeature)) {
      return new Stair(cellMessage.getX(), cellMessage.getY(), Stair.Direction.UP);
    } else if (downStairsSet.contains(dungeonFeature)) {
      return new Stair(cellMessage.getX(), cellMessage.getY(), Stair.Direction.DOWN);
    }
    throw new IllegalArgumentException("cell type not  done yet");
  }

}
