package com.spbsu.crawl.bl;

import com.spbsu.commons.math.vectors.Vec;

import java.util.stream.Stream;

/**
 * Experts League
 * Created by solar on 24/03/16.
 */
public interface Mob {
  Vec position();
  Stream<Action> actions();

  enum Action {
    MOVE_UP('k'),
    MOVE_DOWN('j'),
    MOVE_LEFT('h'),
    MOVE_RIGHT('l'),
    MOVE_UP_RIGHT('u'),
    MOVE_UP_LEFT('y'),
    MOVE_DOWN_RIGHT('n'),
    MOVE_DOWN_LEFT('b'),
    EAT_FROM_FLOOR('e'),
    CHOP_CORPSE('c'),
    INSPECT_TILE(';'),
    PICK_ITEM(','),
    WAIT_A_TURN('.'),
    REST('5'),
    GO_UP('<'),
    GO_DOWN('>'),
    ;


    private char code;
    Action(char code) {
      this.code = code;
    }

    public char code() {
      return code;
    }
  }
}
