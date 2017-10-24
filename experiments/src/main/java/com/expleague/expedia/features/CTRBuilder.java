package com.expleague.expedia.features;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.VecBuilder;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.JsonFeatureMeta;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.node.ObjectNode;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.*;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class CTRBuilder<T> {
  private JsonFeatureMeta meta = new JsonFeatureMeta();
  private TObjectIntHashMap<T> alpha = new TObjectIntHashMap<>();
  private TObjectIntHashMap<T> beta = new TObjectIntHashMap<>();

  private final VecBuilder ctr = new VecBuilder();

  private CTRBuilder() {
  }

  public CTRBuilder(final String id, final String description) {
    meta.id = id;
    meta.description = description;
    meta.type = FeatureMeta.ValueType.VEC;
  }

  public Vec build() {
    return ctr.build();
  }

  public void add(final T key, final boolean alpha) {
    if (alpha) {
      addAlpha(key);
    } else {
      addBeta(key);
    }
  }

  public JsonFeatureMeta getMeta() {
    return meta;
  }

  public void write(final String path) throws IOException {
    writeMeta(this.meta, path + this.meta.id + ".meta.json");
    writeHashMap(this.alpha, path + this.meta.id + ".alpha.gz");
    writeHashMap(this.beta, path + this.meta.id + ".beta.gz");
  }

  public static <T> CTRBuilder<T> load(final String path, final String id) throws IOException, ClassNotFoundException {
    final CTRBuilder<T> builder = new CTRBuilder<>();
    builder.meta = readMeta(path + id + ".meta.json");
    builder.alpha = readHashMap(path + id + ".alpha.gz");
    builder.beta = readHashMap(path + id + ".beta.gz");
    return builder;
  }

  private void addAlpha(final T key) {
    alpha.adjustOrPutValue(key, 1, 1);
    ctr.append(getCTR(key));
  }

  private void addBeta(final T key) {
    beta.adjustOrPutValue(key, 1, 1);
    ctr.append(getCTR(key));
  }

  private double getCTR(final T key) {
    return (alpha.get(key) + 1.0) / (alpha.get(key) + beta.get(key) + 2.0);
  }

  private static void writeMeta(final JsonFeatureMeta meta, final String fileName) throws IOException {
    final OutputStreamWriter out = new OutputStreamWriter(new FileOutputStream(fileName));

    // configure json factory
    final JsonFactory jsonFactory = new JsonFactory();
    jsonFactory.configure(JsonParser.Feature.ALLOW_COMMENTS, false);

    // create mapper
    final ObjectMapper mapper = new ObjectMapper(jsonFactory);
    final ObjectNode output = mapper.createObjectNode();

    final ObjectNode node = output.putObject("meta");
    node.put("id", meta.id());
    node.put("description", meta.description());
    node.put("type", meta.type().toString());

    final ObjectWriter writer = mapper.writerWithDefaultPrettyPrinter();
    out.append(writer.writeValueAsString(output));
    out.close();
  }

  private static JsonFeatureMeta readMeta(final String fileName) throws IOException {
    final InputStreamReader in = new InputStreamReader(new FileInputStream(fileName));
    String json = new BufferedReader(in).lines().parallel().collect(Collectors.joining("\n"));

    // configure json factory
    final JsonFactory jsonFactory = new JsonFactory();
    jsonFactory.configure(JsonParser.Feature.ALLOW_COMMENTS, false);

    // create mapper
    final ObjectMapper mapper = new ObjectMapper(jsonFactory);
    final ObjectReader reader = mapper.reader();
    final JsonNode node = reader.readTree(json);

    final JsonFeatureMeta meta = new JsonFeatureMeta();
    meta.id = node.get("meta").get("id").asText();
    meta.description = node.get("meta").get("description").asText();
    meta.type = FeatureMeta.ValueType.valueOf(node.get("meta").get("type").asText());

    return meta;
  }

  private static <T> void writeHashMap(final TObjectIntHashMap<T> hashMap, final String fileName) throws IOException {
    final ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(fileName)));
    out.writeObject(hashMap);
    out.close();
  }

  private static <T> TObjectIntHashMap<T> readHashMap(final String fileName) throws IOException, ClassNotFoundException {
    final ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(fileName)));
    return (TObjectIntHashMap<T>) in.readObject();
  }
}
