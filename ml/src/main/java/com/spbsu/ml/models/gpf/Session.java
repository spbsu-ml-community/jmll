package com.spbsu.ml.models.gpf;

import gnu.trove.list.array.TIntArrayList;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * User: irlab
 * Date: 14.05.14
 */
public class Session<Blk extends Session.Block> {
  public static final int Q_INDEX = 0;
  public static final int S_INDEX = 1;
  public static final int E_INDEX = 2;
  public static final int R0_INDEX = 3;

  public enum BlockType {
    RESULT, Q, S, E
  }

  public static class Block {
    public final BlockType blockType;
    public final int position;

    public Block(final BlockType blockType, final int position) {
      this.blockType = blockType;
      this.position = position;
    }

    @Override
    public String toString() {
      return "Block{" + blockType +
              ", position=" + position +
              '}';
    }
  }

  public static class Edge {
    public final int block_index_from; // position of block in blocks array
    public final int block_index_to;   // position of block in blocks array

    public Edge(final int block_index_from, final int block_index_to) {
      this.block_index_from = block_index_from;
      this.block_index_to = block_index_to;
    }
  }

  private Blk[] blocks;
  private int[][] edge_from; // there is an Edge blocks[i] -> blocks[edge_from[i][j]]
  private int[][] edge_to;   // there is an Edge blocks[edge_from[i][j]] -> blocks[i]
  private int[] click_indexes;

  private String uid;
  private String reqid;
  private int user_region;
  private String query;
  private String source_string;

  public long timestamp;

  public Session() {
  }

  public Session(final String uid, final String reqid, final int user_region, final String query, final String source_string) {
    this.uid = uid;
    this.reqid = reqid;
    this.user_region = user_region;
    this.query = query;
    this.source_string = source_string;
  }

  public void setBlocks(final Blk[] blocks) {
    this.blocks = blocks;
  }
  public Blk getBlock(final int index) {
    return blocks[index];
  }

  public Blk[] getBlocks() {
    return blocks;
  }

  public int getBlocksCount() {
    return blocks.length;
  }

  public int[] getClick_indexes() {
    return click_indexes;
  }

  public void setClick_indexes(final int[] click_indexes) {
    this.click_indexes = click_indexes;
  }

  public boolean hasClickOn(final int block_index) {
    for (final int i: click_indexes) {
      if (i == block_index) return true;
    }
    return false;
  }

  public int[] getEdgesFrom(final int block_index_from) {
    return edge_from[block_index_from];
  }
  public int[] getEdgesTo(final int block_index_to) {
    return edge_to[block_index_to];
  }

  public void setEdges(final List<Edge> edges) {
    @SuppressWarnings("unchecked") final
    List<Integer>[] edge_from_v = (List<Integer>[])new List[blocks.length];
    @SuppressWarnings("unchecked") final
    List<Integer>[] edge_to_v = (List<Integer>[])new List[blocks.length];
    for (int i = 0; i < blocks.length; i++) {
      edge_from_v[i] = new ArrayList<Integer>();
      edge_to_v[i] = new ArrayList<Integer>();
    }
    for (final Edge e: edges) {
      edge_from_v[e.block_index_from].add(e.block_index_to);
      edge_to_v[e.block_index_to].add(e.block_index_from);
    }

    edge_from = new int[blocks.length][];
    edge_to = new int[blocks.length][];
    for (int i = 0; i < blocks.length; i++) {
      edge_from[i] = new int[edge_from_v[i].size()];
      for (int j = 0; j < edge_from[i].length; j++)
        edge_from[i][j] = (Integer)edge_from_v[i].get(j);
      Arrays.sort(edge_from[i]);

      edge_to[i] = new int[edge_to_v[i].size()];
      for (int j = 0; j < edge_to[i].length; j++)
        edge_to[i][j] = (Integer)edge_to_v[i].get(j);
      Arrays.sort(edge_to[i]);
    }
  }

  public void sortUniqueClicks() {
    if (click_indexes.length == 0) return;
    Arrays.sort(click_indexes);
    final TIntArrayList new_click_indexes = new TIntArrayList(click_indexes.length);
    new_click_indexes.add(click_indexes[0]);
    for (int i = 1; i < click_indexes.length; i++) {
      if (click_indexes[i] == click_indexes[i-1]) continue;
      new_click_indexes.add(click_indexes[i]);
    }
    if (new_click_indexes.size() == click_indexes.length)
      return;
    click_indexes = new_click_indexes.toArray();
  }

  public String getUid() {
    return uid;
  }

  public String getReqid() {
    return reqid;
  }

  public int getUser_region() {
    return user_region;
  }

  public String getQuery() {
    return query;
  }

  public String getSource_string() {
    return source_string;
  }

  @Override
  public String toString() {
    return "Session{" +
           "source_string='" + source_string + '\'' +
            ",\nclick_indexes=" + Arrays.toString(click_indexes) +
            '}';
  }
}
