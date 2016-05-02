package com.spbsu.crawl.data.impl;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public enum KeyCode {
  UP(-254),
  DOWN(-253),
  LEFT(-252),
  RIGHT(-251),
  a(65),
  b(66),
  c(67),
  d(68),
  e(69),
  f(70),
  g(71),
  h(72),
  i(73),
  j(74),
  k(75),
  l(76),
  m(77),
  n(78),
  o(79),
  p(80),
  q(81),
  r(82),
  s(83),
  t(84),
  u(85),
  v(86),
  w(87),
  x(88),
  y(89),
  z(90),
  ENTER(13),
  ESCAPE(27),
  ;

  private final int code;

  KeyCode(int code) {
    this.code = code;
  }

  public int getCode() {
    return code;
  }
}
