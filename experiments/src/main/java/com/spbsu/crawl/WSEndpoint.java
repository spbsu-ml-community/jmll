package com.spbsu.crawl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.seq.CharSequenceReader;
import com.spbsu.crawl.data.Command;
import com.spbsu.crawl.data.Message;
import com.spbsu.crawl.data.Protocol;

import javax.websocket.*;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
@ClientEndpoint
public class WSEndpoint {
  private final Thread outThread;
  private BlockingQueue<Message> in = new LinkedBlockingQueue<>();
  private BlockingQueue<Message> out = new LinkedBlockingQueue<>();
  private final Session session;
  private Inflater inflater;
  private CharsetDecoder decoder;
  private final ObjectMapper mapper;

  public WSEndpoint(URI uri) throws IOException, DeploymentException {
    final WebSocketContainer container = ContainerProvider.getWebSocketContainer();
    mapper = new ObjectMapper();
    session = container.connectToServer(this, uri);
    decoder = StreamTools.UTF.newDecoder();
    inflater = new Inflater(true);

    outThread = new Thread(() -> {
      try {
        //noinspection InfiniteLoopStatement
        while (true) {
          final Message poll = out.take();
          final ObjectNode node = mapper.valueToTree(poll);
          node.set("msg", new TextNode(poll.type().name().toLowerCase()));
          final String clientJson = mapper.writeValueAsString(node);
          session.getAsyncRemote().sendText(clientJson);
          System.out.println("[CLIENT]: " + clientJson);
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
    System.out.println("Closed");
    inflater.end();
  }

  @OnError
  public void error(Throwable th) {
    th.printStackTrace();
    System.out.println("Error");
    inflater.end();
  }

  @OnMessage
  public void onMessage(String msg) {
    handleMessage(new StringReader(msg));
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
      inflater.setInput(inBuffer.array(), inBuffer.position(), inBuffer.remaining());
      final StringBuilder builder = new StringBuilder();
      final ByteBuffer outBuffer = ByteBuffer.allocate(4096);
      while (!inflater.needsInput()) {
        final int inflate = inflater.inflate(outBuffer.array(), outBuffer.position(), outBuffer.remaining());
        outBuffer.limit(outBuffer.position() + inflate);
        builder.append(decoder.decode(outBuffer));
        outBuffer.compact();
      }
      handleMessage(new CharSequenceReader(builder));
    } catch (DataFormatException | CharacterCodingException e) {
      throw new RuntimeException(e);
    }
  }

  private void handleMessage(final Reader messageReader) {
    final JsonNode node;
    try {
      node = mapper.readTree(messageReader);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    final JsonNode msgs = node.get("msgs");
    System.out.println("[SERVER]: " + node.toString());
    if (msgs != null) {
      for (JsonNode msg : msgs) {
        onItem(msg);
      }
    } else {
      onItem(node);
    }
  }

  private void onItem(JsonNode node) {
    try {
      final Protocol msg = Protocol.valueOf(node.get("msg").asText().toUpperCase());
      final Message message = mapper.treeToValue(node, msg.clazz());
      if (message instanceof Command) {
        ((Command) message).execute(out);
      }
      else {
        in.put(message);
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  @OnOpen
  public void open(Session wsSession) {
  }

  public void send(Message message) {
    out.add(message);
  }

  public BlockingQueue<Message> getMessagesQueue() {
    return in;
  }
}
