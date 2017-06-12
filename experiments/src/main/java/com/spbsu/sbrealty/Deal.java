package com.spbsu.sbrealty;

import com.spbsu.ml.meta.DSItem;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import java.util.Date;

/**
 * Experts League
 * Created by solar on 05.06.17.
 */
@XmlRootElement
public class Deal implements DSItem {
  @XmlAttribute
  private String id;

  @XmlAttribute
  private double area;
  @XmlAttribute
  private double larea;

  @XmlAttribute
  private int floor;
  @XmlAttribute
  private int maxFloor;

  @XmlAttribute
  private int roomsCount;

  @XmlAttribute
  private double kitchenArea;

  @XmlAttribute
  private int type;
  @XmlAttribute
  private long dateTs;
  @XmlAttribute
  private String district;


  public double area() {
    return area;
  }

  public double larea() {
    return larea;
  }

  public int floor() {
    return floor;
  }

  public int maxFloor() {
    return maxFloor;
  }

  public int roomsCount() {
    return roomsCount;
  }

  public double kitchenArea() {
    return kitchenArea;
  }

  public int type() {
    return type;
  }

  public long dateTs() {
    return dateTs;
  }

  public String district() {
    return district;
  }

  @Override
  public String id() {
    return id;
  }

  public static class Builder {
    private Deal result = new Deal();

    public Builder() {}
    public Builder(Deal deal) {
      result = deal;
    }

    public Deal build() {
      final Deal deal = this.result;
      this.result = new Deal();
      return deal;
    }

    public void id(String id) {
      result.id = id;
    }

    public void area(double full_sq) {
      result.area = full_sq;
      if (result.larea == 0)
        result.larea = full_sq / 2;
    }

    public void livingArea(double v) {
      result.larea = v;
    }

    public void floor(int floor) {
      result.floor = floor;
    }

    public void maxFloor(int maxFloor) {
      result.maxFloor = maxFloor;
    }

    public void roomsCount(int num_room) {
      result.roomsCount = num_room;
    }

    public void kitchenArea(double kitch_sq) {
      result.kitchenArea = kitch_sq;
    }

    public void type(int material) {
      result.type = material;
    }

    public void date(Date date) {
      result.dateTs = date.getTime();
    }

    public void district(String district) {
      result.district = district;
    }

    public boolean valid() {
      return result.type >= 0 && result.floor <= result.maxFloor && result.area > result.larea && result.larea > 0;
    }

    public boolean semiValid() {
      return (result.floor <= result.maxFloor || result.maxFloor < 0) && result.area > result.larea && result.larea > 0;
    }

    public boolean quiteNotValid() {
      return (result.floor <= result.maxFloor || result.maxFloor < 0) || result.area > result.larea || result.larea > 0;
    }
  }
}
