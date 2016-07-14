package com.spbsu.crawl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.seq.CharSequenceReader;
import com.spbsu.crawl.data.Message;
import com.spbsu.crawl.data.Protocol;

import javax.websocket.*;
import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
@ClientEndpoint
public class WSEndpoint {
  private static final Logger log = Logger.getLogger(WSEndpoint.class.getName());
  @SuppressWarnings("FieldCanBeLocal")
  private final Thread outThread;
  private BlockingQueue<Message> in = new LinkedBlockingQueue<>();
  BlockingQueue<Message> out = new LinkedBlockingQueue<>();
  private final Session session;
  private Inflater inflater;
  private CharsetDecoder decoder;
  private final ObjectMapper mapper;

  public WSEndpoint(URI uri) throws IOException, DeploymentException {
    decoder = StreamTools.UTF.newDecoder();
    mapper = new ObjectMapper();
    inflater = new Inflater(true);

    final WebSocketContainer container = ContainerProvider.getWebSocketContainer();
    session = container.connectToServer(this, uri);

    outThread = new Thread(() -> {
      try {
        //noinspection InfiniteLoopStatement
        while (true) {
          final Message poll = out.take();
          final ObjectNode node = mapper.valueToTree(poll);
          node.set("msg", new TextNode(poll.type().name().toLowerCase()));
          final String clientJson = mapper.writeValueAsString(node);
          session.getAsyncRemote().sendText(clientJson);
//          log.info("[CLIENT]: " + clientJson);
        }
      } catch (InterruptedException | JsonProcessingException e) {
        throw new RuntimeException(e);
      }
    }, "JSON Output thread");
    outThread.setDaemon(true);
    outThread.start();
  }

  @OnClose
  public void close() {
    log.info("Closed");
    inflater.end();
  }

  @OnError
  public void error(Throwable th) {
    th.printStackTrace();
    log.log(Level.WARNING, "Error", th);
    inflater.end();
  }

  @OnMessage
  public void onMessage(String msg) {
    handleMessage(msg);
  }

  @OnMessage
  public void onMessage(ByteBuffer buffer) {
    try {
      final ByteBuffer inBuffer = ByteBuffer.allocate(buffer.remaining() + 4);
      inBuffer.put(buffer);
      inBuffer.put((byte) 0);
      inBuffer.put((byte) 0);
      inBuffer.put((byte) -1);
      inBuffer.put((byte) -1);
      inBuffer.flip();
      if (!inBuffer.hasArray()) {
        System.out.println("Error: message with empty buffer");
        return;
      }
      inflater.setInput(inBuffer.array(), inBuffer.position(), inBuffer.remaining());
      final StringBuilder builder = new StringBuilder();
      final ByteBuffer outBuffer = ByteBuffer.allocate(4096);
      while (!inflater.needsInput()) {
        final int inflate = inflater.inflate(outBuffer.array(), outBuffer.position(), outBuffer.remaining());
        outBuffer.limit(outBuffer.position() + inflate);
        builder.append(decoder.decode(outBuffer));
        outBuffer.compact();
      }
      handleMessage(builder);
    }
    catch (DataFormatException | CharacterCodingException e) {
      log.log(Level.WARNING, "WS message format exception: " + e.getMessage());
    }
  }

  private void handleMessage(final CharSequence message) {
    final JsonNode node;
    try {
      node = mapper.readTree(new CharSequenceReader(message));
    }
    catch (IOException e) {
//      log.log(Level.WARNING, "JSON format exception. Message: '" + message + "'. Exception msg: " + e.getMessage() + ". Skipping the message.");
      return;
    }

    final JsonNode msgs = node.get("msgs");
    log.info("[SERVER]: " + node.toString());
    if (msgs != null) {
      for (JsonNode msg : msgs) {
        onItem(msg);
      }
    } else {
      onItem(node);
    }
  }

  public Message poll() {
    try {
      return in.poll(1, TimeUnit.HOURS);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  public void send(Message msg) {
    try {
      out.put(msg);
    }
    catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  private void onItem(JsonNode node) {
    try {
      final Protocol msg = Protocol.valueOf(node.get("msg").asText().toUpperCase());
      in.put(mapper.treeToValue(node, msg.clazz()));
    }
    catch (JsonProcessingException | InterruptedException e) {
      throw new RuntimeException(e);
    }
    catch (IllegalArgumentException iae) {
      log.warning("Unknown message type: " + node.get("msg").asText().toUpperCase());
    }
  }

  @OnOpen
  public void open(Session wsSession) {
  }

  public BlockingQueue<Message> getMessagesQueue() {
    return in;
  }
}
