package com.spbsu.ml.models.gpf;

import com.spbsu.commons.filters.FalseFilter;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * User: irlab
 * Date: 14.05.14
 */
public class Session {
  static final int Q_ind = 0;
  static final int S_ind = 1;
  static final int E_ind = 2;
  static final int R0_ind = 3;

  public static enum BlockType {
    RESULT, Q, S, E
  }
  public static enum ResultType {
    WEB, NEWS, IMAGES, DIRECT, VIDEO, OTHER
  }
  public static enum ResultGrade {
    VITAL           (0.61), //0.69),
    USEFUL          (0.41), //0.47),
    RELEVANT_PLUS   (0.14), //0.45),
    RELEVANT_MINUS  (0.07), //0.44),
    IRRELEVANT      (0.03), //0.24),
    NOT_ASED        (0.10); //0.39);

    private final double pfound_value;
    //private final double ctr1_value;
    ResultGrade(double pfound_value) {
      this.pfound_value = pfound_value;
    }

    public double getPfound_value() {
      return pfound_value;
    }
  }
  public static class Block {
    BlockType blockType;
    ResultType resultType;
    int position;
    ResultGrade resultGrade;

    public Block(BlockType blockType, int position) {
      this.blockType = blockType;
      this.position = position;
    }

    public Block(BlockType blockType, ResultType resultType, int position, ResultGrade resultGrade) {
      this.blockType = blockType;
      this.resultType = resultType;
      this.position = position;
      this.resultGrade = resultGrade;
    }

    @Override
    public String toString() {
      return "Block{" + blockType +
              ", " + resultType +
              ", " + resultGrade +
              ", position=" + position +
              '}';
    }
  }
  public static class Edge {
    int block_index_from; // position of block in blocks array
    int block_index_to;   // position of block in blocks array
//    float[] feats;

    public Edge(int block_index_from, int block_index_to) {
      this.block_index_from = block_index_from;
      this.block_index_to = block_index_to;
    }
  }

  private Block[] blocks;
  private int[][] edge_from; // there is an Edge blocks[i] -> blocks[edge_from[i][j]]
  private int[][] edge_to;   // there is an Edge blocks[edge_from[i][j]] -> blocks[i]
  private int[] click_indexes;

  private String uid;
  private String reqid;
  private int user_region;
  private String query;
  private String source_string;

  public Session() {
  }

  public Session(String uid, String reqid, int user_region, String query, String source_string) {
    this.uid = uid;
    this.reqid = reqid;
    this.user_region = user_region;
    this.query = query;
    this.source_string = source_string;
  }

  public void setBlocks(Block[] blocks) {
    this.blocks = blocks;
  }
  public Block getBlock(int index) {
    return blocks[index];
  }
  public Block[] getBlocks() {
    return blocks;
  }

  public int[] getClick_indexes() {
    return click_indexes;
  }

  public void setClick_indexes(int[] click_indexes) {
    this.click_indexes = click_indexes;
  }

  public boolean hasClickOn(int block_index) {
    for (int i: click_indexes) {
      if (i == block_index) return true;
    }
    return false;
  }

  public int[] getEdgesFrom(int block_index_from) {
    return edge_from[block_index_from];
  }
  public int[] getEdgesTo(int block_index_to) {
    return edge_to[block_index_to];
  }

  public void setEdges(List<Edge> edges) {
    List<Integer>[] edge_from_v = (List<Integer>[])new List[blocks.length];
    List<Integer>[] edge_to_v = (List<Integer>[])new List[blocks.length];
    for (int i = 0; i < blocks.length; i++) {
      edge_from_v[i] = new ArrayList<Integer>();
      edge_to_v[i] = new ArrayList<Integer>();
    }
    for (Edge e: edges) {
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
    TIntArrayList new_click_indexes = new TIntArrayList(click_indexes.length);
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
