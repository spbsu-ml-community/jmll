package com.expleague.expedia;

import com.expleague.ml.meta.DSItem;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

@XmlRootElement
public class EventItem extends DSItem.Stub {
  @XmlAttribute
  private int day;
  @XmlAttribute
  private int user;
  @XmlAttribute
  private int hotel;

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
