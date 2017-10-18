package com.expleague.crawl.bl.map;

import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.stream.Stream;

/**
 * User: Noxoomo
 * Date: 20.08.16
 * Time: 22:10
 */

public class PositionManager {

  private int coordinateToInt(int x, int y) {
    final int MAX_AXIS_LENGTH = 10000;
    return x + MAX_AXIS_LENGTH * y;
  }

  private TIntObjectHashMap<PositionImpl> knownPositions = new TIntObjectHashMap<>();

  private PositionImpl create(final int x, final int y) {
    final PositionImpl position = new PositionImpl(x, y);
    knownPositions.put(coordinateToInt(x, y), position);
    return position;
  }

  public Position getOrCreate(final int x, final int y) {
    final int coordinateDef = coordinateToInt(x, y);
    if (!knownPositions.containsKey(coordinateDef)) {
      return create(x, y);
    }
    return knownPositions.get(coordinateDef);
  }

  public void moveCenter(final int x, final int y) {
    final TIntObjectHashMap<PositionImpl> movedKnownPositions = new TIntObjectHashMap<>();
    final Stream<PositionImpl> positions = knownPositions.valueCollection().stream().map(position -> position.moveCenter(x, y));
    positions.forEach(position -> movedKnownPositions.put(coordinateToInt(position.x, position.y), position));
    knownPositions = movedKnownPositions;
  }

  public void clear() {
    knownPositions.clear();
  }


  //for dist and other: we should calc everything about position through Position class.
  //we can't calc anything for position with different managers. (it could be different levels, etc)
//  private PositionManager ptr() {
//    return this;
//  }
  public class PositionImpl implements Position {
    private int x;
    private int y;

    protected PositionImpl(final int x,
                           final int y) {
      this.x = x;
      this.y = y;
    }

    protected PositionImpl moveCenter(final int x, final int y) {
      this.x -= x;
      this.y -= y;
      return this;
    }

    @Override
    public int x() {
      return x;
    }

    @Override
    public int y() {
      return y;
    }

    @Override
    public boolean equals(final Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;

      final PositionImpl position = (PositionImpl) o;

      if (x != position.x) return false;
      return y == position.y;

    }

    @Override
    public int hashCode() {
      int result = x;
      result = 31 * result + y;
      return result;
    }
  }
}

