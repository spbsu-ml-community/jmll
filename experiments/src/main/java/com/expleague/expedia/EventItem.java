package com.expleague.expedia;

import com.expleague.ml.meta.DSItem;

public class EventItem implements DSItem {
  public int day;
  public int user;
  public int hotel;

  public EventItem() {

  }

  public EventItem(final int day, final int user, final int hotel) {
    this.day = day;
    this.user = user;
    this.hotel = hotel;
  }

  @Override
  public String id() {
    return user + "@" + hotel + ":";
  }
}
