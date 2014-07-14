package com.spbsu.ml.meta.items;

import com.spbsu.ml.meta.DSItem;

/**
 * User: solar
 * Date: 11.07.14
 * Time: 21:25
 */
public class QURLItem implements DSItem {
  public final int queryId;
  public final String url;
  public final int groupId;

  public QURLItem(int queryId, String url, int groupId) {
    this.queryId = queryId;
    this.url = url;
    this.groupId = groupId;
  }

  @Override
  public String id() {
    return url + "@" + queryId;
  }
}
