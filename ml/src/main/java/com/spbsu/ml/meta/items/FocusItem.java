package com.spbsu.ml.meta.items;

import com.spbsu.ml.meta.DSItem;


public class FocusItem implements DSItem {
  public String uuid;
  public String appId;
  public long ts;


  public FocusItem() {
  }

  public FocusItem(final String uuid, final String appId, final long ts) {
    this.uuid = uuid;
    this.appId = appId;
    this.ts = ts;
  }

  public String getUuid() {
    return uuid;
  }

  public void setUuid(final String uuid) {
    this.uuid = uuid;
  }

  public String getAppId() {
    return appId;
  }

  public void setAppId(final String appId) {
    this.appId = appId;
  }

  public long getTs() {
    return ts;
  }

  public void setTs(final long ts) {
    this.ts = ts;
  }

  @Override
  public String id() {
    return String.format("%s:%s:%d", uuid, appId, ts);
  }
}
