package com.spbsu.ml.models.gpf.weblogmodel;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import com.google.gson.*;
import com.spbsu.ml.models.gpf.GPFLinearModel;
import com.spbsu.ml.models.gpf.GPFModel;
import com.spbsu.ml.models.gpf.Session;
import com.spbsu.ml.models.gpf.weblogmodel.BlockV1;
import gnu.trove.list.array.TIntArrayList;

/**
 * User: irlab
 * Date: 22.05.14
 */
public class WebLogV1GPFSession {
  public static List<Session<BlockV1>> loadDatasetFromJSON(InputStream is, GPFModel model, int rows_limit) throws IOException {
    List<Session<BlockV1>> dataset = new ArrayList<>();

    LineNumberReader lnr = new LineNumberReader(new InputStreamReader(is, "UTF8"));
    Gson gson = new Gson();
    Gson gson_prettyprint = new GsonBuilder().setPrettyPrinting().create();
    for (String line = lnr.readLine(); line != null; line = lnr.readLine()) {
      if (rows_limit > 0 && dataset.size() >= rows_limit)
        break;

      String[] split = line.split("\t");
      String json_ses_str = split[2];

      JsonSes ses = gson.fromJson(json_ses_str, JsonSes.class);

      BlockV1[] blocks = new BlockV1[ses.sntypes.length];
      for (int i = 0; i < blocks.length; i++) {
        blocks[i] = new BlockV1(
                Session.BlockType.RESULT,
                BlockV1.ResultType.valueOf(ses.sntypes[i]),
                i,
                BlockV1.ResultGrade.valueOf(ses.rel[i]));
      }

      String source_string = gson_prettyprint.toJson(ses);
      Session<BlockV1> session = new Session<BlockV1>(ses.uid, ses.reqid, ses.user_region, ses.query, source_string);
      setSessionData(session, blocks, ses.clicks);
      dataset.add(session);
    }
    return dataset;
  }

  /**
   * this function sets up the structure of a Session: a set of vertices (blocks and virtual blocks), and a set of edges
   * @param ses - a Session to set up (write-only)
   * @param result_blocks - a set of 'real' (observed) blocks (read-only)
   * @param clicks_block_indexes - list of clicks (clicks_block_indexes[i] is a i'th click on result_blocks[clicks_block_indexes[i]])
   */
  public static void setSessionData(Session<BlockV1> ses, BlockV1[] result_blocks, int[] clicks_block_indexes) {
    // init blocks
    BlockV1[] blocks = new BlockV1[result_blocks.length + Session.R0_ind];
//    int[] result_pos2block_ind = new int[100];
    int max_result_pos = -1;
    int min_result_pos = 1000;

    blocks[Session.Q_ind] = new BlockV1(Session.BlockType.Q, -1);
    blocks[Session.S_ind] = new BlockV1(Session.BlockType.S, -1);
    blocks[Session.E_ind] = new BlockV1(Session.BlockType.E, -1);
    for (int i = 0; i < result_blocks.length; i++) {
      blocks[i + Session.R0_ind] = result_blocks[i];
      max_result_pos = Math.max(max_result_pos, result_blocks[i].position);
      min_result_pos = Math.min(min_result_pos, result_blocks[i].position);
    }
    ses.setBlocks(blocks);

    int[] click_indexes = new int[clicks_block_indexes.length];
    for (int i = 0; i < click_indexes.length; i++)
      click_indexes[i] = clicks_block_indexes[i] + Session.R0_ind;
    ses.setClick_indexes(click_indexes);

    // init edges
    List<Session.Edge> edges = new ArrayList<>();
    for (int i = Session.R0_ind; i < blocks.length; i++) {
      // R_i -> R_{i+1}
      if (i + 1 < blocks.length)
        edges.add(new Session.Edge(i, i+1));
      // R_i -> R_{i-1}
      if (i > Session.R0_ind)
        edges.add(new Session.Edge(i, i-1));
      // Q -> R_i
      edges.add(new Session.Edge(Session.Q_ind, i));
      // S -> R_i
      edges.add(new Session.Edge(Session.S_ind, i));
      // R_i -> S
      edges.add(new Session.Edge(i, Session.S_ind));
      // R_i -> E
      edges.add(new Session.Edge(i, Session.E_ind));
      // E -> E
      edges.add(new Session.Edge(Session.E_ind, Session.E_ind));
    }
    ses.setEdges(edges);
  }


  //  {
  //    "__module__": "gpf_learn",
  //    "uid": "y1495086881377522622",
  //    "timestamp": 1377789431,
  //    "rel": ["RELEVANT_PLUS", "RELEVANT_PLUS", "RELEVANT_PLUS", "NOT_ASED", "NOT_ASED", "RELEVANT_PLUS", "RELEVANT_PLUS", "RELEVANT_PLUS", "RELEVANT_MINUS", "NOT_ASED", "RELEVANT_PLUS"],
  //    "reqid": "1377789430798872-1404107670513402777522981-ws38-714",
  //    "user_region": 213,
  //    "query": "строение человека",
  //    "__name__": "Session",
  //    "sntypes": ["WEB", "WEB", "WEB", "IMAGES", "WEB", "WEB", "WEB", "WEB", "WEB", "WEB", "WEB"],
  //    "clicks": [0, 3],
  //    "clicks_dwelltime": [3, 2]
  //  }
  public static class JsonSes {
    String __module__;
    String uid;
    long timestamp;
    String[] rel;
    String reqid;
    int user_region;
    String query;
    String __name__;
    String[] sntypes;
    int[] clicks;
    long[] clicks_dwelltime;

    public JsonSes() {
    }
  }

  public static void main(String[] args) throws IOException {
    // test
    long t1 = System.currentTimeMillis();
    List<Session<BlockV1>> dataset = loadDatasetFromJSON(new GZIPInputStream(WebLogV1GPFSession.class.getResourceAsStream("ses_100k_simple_rand1.dat.gz")), new GPFLinearModel(), 0);
    System.out.println("dataset size: " + dataset.size());
    System.out.println("time: " + (System.currentTimeMillis() - t1));
    System.out.println("dataset[0]: " + dataset.get(0));
  }
}
