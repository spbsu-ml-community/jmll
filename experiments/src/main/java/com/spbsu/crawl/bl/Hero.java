package com.spbsu.crawl.bl;

/**
 * Experts League
 * Created by solar on 21/04/16.
 */
public class Hero {

  public enum Stat {
    Strength('S'),
    Intellegence('I'),
    Dexterity('D'),
    ;

    private final char select_role;
    Stat(char code) {
      select_role = code;
    }

    public char select() {
      return select_role;
    }
  }

  public enum Spec {
    Fighter_Unarmed('a', 'a'),
    Fighter_Axe('a', 'd'),
    ;

    private final char select_role;
    private final Character select_spec;
    Spec(char code, Character spec) {
      select_role = code;
      select_spec = spec;
    }

    public char selectProf() {
      return select_role;
    }

    public char selectSpec() {
      return select_spec;
    }

    public boolean hasSpec() {
      return select_spec != null;
    }
  }

  public enum Race {
    Minotaur('n');

    private final char select;
    Race(char code) {
      select = code;
    }

    public char select() {
      return select;
    }
  }
}
