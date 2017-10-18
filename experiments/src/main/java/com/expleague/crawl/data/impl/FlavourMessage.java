package com.expleague.crawl.data.impl;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.expleague.crawl.data.Message;

/**
 * User: Noxoomo
 * Date: 07.04.16
 * Time: 18:09
 */

// Это добро шлется с конвертацией в лонги. Все очень приятно для разбора…
//  static inline void _write_tileidx(tileidx_t t)
//  {
//    // JS can only handle signed ints
//    const int lo = t & 0xFFFFFFFF;
//    const int hi = t >> 32;
//    if (hi == 0)
//      tiles.write_message("%d", lo);
//    else
//      tiles.write_message("[%d,%d]", lo, hi);
//  }

public class FlavourMessage implements Message {

  @JsonProperty("f")
  private int f;

  @JsonProperty("s")
  private int s;
}

