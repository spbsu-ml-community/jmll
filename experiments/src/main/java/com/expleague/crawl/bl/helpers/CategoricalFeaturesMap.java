package com.expleague.crawl.bl.helpers;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import gnu.trove.map.hash.TObjectIntHashMap;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by noxoomo on 14/07/16.
 */
public class CategoricalFeaturesMap {
  private static final Logger log = Logger.getLogger(CategoricalFeaturesMap.class.getName());

  private TObjectIntHashMap<String> direct = new TObjectIntHashMap<>();
  private List<String> inverse = new ArrayList<>();

  public int value(@NotNull String message) {
    if (direct.containsKey(message)) {
      return direct.get(message);
    } else {
      return -1;
    }
  }

  public String value(int id) {
    if (id >= 0 && id < inverse.size()) {
      return inverse.get(id);
    }
    return null;
  }

  public int dictSize() {
    return inverse.size();
  }


  public void save(final File file) {

    ObjectMapper mapper = new ObjectMapper();
    ObjectNode node = mapper.createObjectNode();
    ArrayNode jsonList = mapper.createArrayNode();
    inverse.forEach(jsonList::add);
    node.set("inverse_index", jsonList);
    try {
      BufferedWriter writer = new BufferedWriter(new FileWriter(file));
      writer.write(new JsonProgress(this).toJson());
      writer.flush();
      writer.close();
    } catch (IOException e) {
      log.log(Level.WARNING, "Can't save cat feature index to file");
      e.printStackTrace();
    }
  }

  public static CategoricalFeaturesMap load(final File file) throws IOException {
    try {
      String data = new BufferedReader(new FileReader(file)).lines().reduce((l, r) -> l + r).get();
      return JsonProgress.fromJson(data);
    } catch (IOException e) {
      log.log(Level.ALL, "Error: can't load cat features map");
      throw e;
    }
  }

  @JsonIgnoreProperties(ignoreUnknown = true)
  private static class JsonProgress {
    @JsonProperty("inverse_index")
    final List<String> inverseIndex;

    public JsonProgress() {
      inverseIndex = new ArrayList<>();
    }

    public JsonProgress(final CategoricalFeaturesMap map) {
      this.inverseIndex = map.inverse;
    }

    public List<String> inverseIndex() {
      return inverseIndex;
    }

    String toJson() {
      ObjectMapper mapper = new ObjectMapper();
      final ObjectNode node = mapper.valueToTree(this);
      try {
        return mapper.writeValueAsString(node);
      } catch (JsonProcessingException e) {
        throw new RuntimeException(e);
      }
    }

    static CategoricalFeaturesMap fromJson(String text) {
      ObjectMapper mapper = new ObjectMapper();
      JsonProgress data;
      try {
        final Class<JsonProgress> clazz = JsonProgress.class;
        data = mapper.readValue(text, clazz);
      } catch (Exception e) {
        throw new RuntimeException(e);
      }

      final CategoricalFeaturesMap map = new CategoricalFeaturesMap();
//      data.inverseIndex().forEach(map::addNewEntryAndReturn);
      return map;
    }
  }
}
