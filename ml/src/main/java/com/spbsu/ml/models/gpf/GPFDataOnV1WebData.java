package com.spbsu.ml.models.gpf;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import com.google.gson.*;

/**
 * User: irlab
 * Date: 22.05.14
 */
public class GPFDataOnV1WebData {
  public static List<Session> loadDatasetFromJSON(String filename, GPFModel model, int rows_limit) throws IOException {
    List<Session> dataset = new ArrayList<Session>();

    InputStream is;
    if (filename.endsWith(".gz")) {
      is = new GZIPInputStream(new FileInputStream(filename));
    } else {
      is = new FileInputStream(filename);
    }
    LineNumberReader lnr = new LineNumberReader(new InputStreamReader(is, "UTF8"));
    Gson gson = new Gson();
    Gson gson_prettyprint = new GsonBuilder().setPrettyPrinting().create();
    for (String line = lnr.readLine(); line != null; line = lnr.readLine()) {
      if (rows_limit > 0 && dataset.size() >= rows_limit)
        break;

      String[] split = line.split("\t");
      String json_ses_str = split[2];

      JsonSes ses = gson.fromJson(json_ses_str, JsonSes.class);

      Session.SessionOnV1WebData.BlockV1[] blocks = new Session.SessionOnV1WebData.BlockV1[ses.sntypes.length];
      for (int i = 0; i < blocks.length; i++) {
        Session.SessionOnV1WebData.BlockV1 block = new Session.SessionOnV1WebData.BlockV1(
                Session.BlockType.RESULT,
                Session.SessionOnV1WebData.ResultType.valueOf(ses.sntypes[i]),
                i,
                Session.SessionOnV1WebData.ResultGrade.valueOf(ses.rel[i]));
        blocks[i] = block;
      }

      String source_string = gson_prettyprint.toJson(ses);
      Session session = new Session.SessionOnV1WebData(ses.uid, ses.reqid, ses.user_region, ses.query, source_string);
      model.setSessionData(session, blocks, ses.clicks);
      dataset.add(session);
    }
    is.close();
    return dataset;
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
    List<Session> dataset = loadDatasetFromJSON("C:\\PRG\\2013\\metric\\data\\20140105_gpf\\f100\\ses_100k_simple_rand1.dat", new GPFLinearModel(), 0);
    System.out.println("dataset size: " + dataset.size());
    System.out.println("time: " + (System.currentTimeMillis() - t1));
    System.out.println("dataset[0]: " + dataset.get(0));
  }
}
