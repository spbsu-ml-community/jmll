package com.spbsu.crawl;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.seq.CharSequenceReader;
import com.spbsu.crawl.data.Command;
import com.spbsu.crawl.data.LoginMessage;
import com.spbsu.crawl.data.Message;

import javax.websocket.*;
import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
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
          session.getAsyncRemote().sendText(mapper.writeValueAsString(node));
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
    System.out.println(">" + msg);
  }

  @OnMessage
  public void onMessage(ByteBuffer buffer) {
//      System.out.println("Binary received");
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
      final JsonNode node = mapper.readTree(new CharSequenceReader(builder));
      final JsonNode msgs = node.get("msgs");
      if (msgs != null) {
        for (JsonNode msg : msgs) {
          onItem(msg);
        }
      } else onItem(node);
      System.out.println(builder.toString());
    } catch (DataFormatException | IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void onItem(JsonNode node) {
    try {
      final Message.Types msg = Message.Types.valueOf(node.get("msg").asText().toUpperCase());
      if (out instanceof Command) {
        ((Command) out).execute(out);
      }
      else {
        in.put(mapper.treeToValue(node, msg.clazz()));
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
}
