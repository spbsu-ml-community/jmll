package com.expleague.ml.meta.items;

import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.GroupedDSItem;

/**
 * User: solar
 * Date: 11.07.14
 * Time: 21:25
 */
public class QURLItem extends FakeItem implements GroupedDSItem {
  public int queryId;
  public String url;
  public int groupId;

  public QURLItem() {
  }

  public QURLItem(final int queryId, final String url, final int groupId) {
    this.queryId = queryId;
    this.url = url;
    this.groupId = groupId;
  }

  @Override
  public String id() {
    return url + "@" + queryId;
  }

  @Override
  public String groupId() {
    return Integer.toString(queryId);
  }

  public String subgroupId() {
    return Integer.toString(groupId);
  }
}
