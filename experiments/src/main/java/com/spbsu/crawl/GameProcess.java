package com.spbsu.crawl;

import com.spbsu.commons.util.ThreadTools;
import com.spbsu.crawl.data.Message;
import com.spbsu.crawl.data.impl.*;
import com.spbsu.crawl.data.impl.system.GoLobbyMessage;
import com.spbsu.crawl.data.impl.system.IgnoreMessage;
import com.spbsu.crawl.data.impl.system.LoginMessage;
import com.spbsu.crawl.data.impl.system.StartGameMessage;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.function.Consumer;
import java.util.logging.Logger;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public class GameProcess {
  private static final Logger LOG = Logger.getLogger(GameProcess.class.getName());

  private final WSEndpoint endpoint;

  public GameProcess(WSEndpoint endpoint) {
    this.endpoint = endpoint;
  }

  public void start() {
    initGame();

    interact(new KeyMessage(KeyCode.LEFT), messages -> System.out.println("messages: " + messages));
    interact(new KeyMessage(KeyCode.RIGHT), messages -> System.out.println("messages: " + messages));
    interact(new KeyMessage(KeyCode.RIGHT), messages -> System.out.println("messages: " + messages));

//    while (true) {
//      ThreadTools.sleep(1000);
//    }
  }

  private void interact(final Message userMessage, final Consumer<List<Message>> handler) {
    endpoint.send(userMessage);
    final BlockingQueue<Message> messagesQueue = endpoint.getMessagesQueue();
    final List<Message> newMessages = new ArrayList<>();

    Message message;
    try {
      while (!InputModeMessage.isEndMessage(message = messagesQueue.take())) {

        if (InputModeMessage.isStartMessage(message) || message instanceof IgnoreMessage) {
          continue;
        }
        newMessages.add(message);
      }
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }

    handler.accept(newMessages);
  }


  private void initGame() {
    endpoint.send(new LoginMessage("asd", "asd"));
    endpoint.send(new GoLobbyMessage());
    ThreadTools.sleep(1000);
    skipMessages();

    interact(new StartGameMessage("dcss-web-trunk"),  messages -> {
      final PlayerInfoMessage playerInfo = findMessage(messages, PlayerInfoMessage.class);
      final MapInfoMessage mapInfo = findMessage(messages, MapInfoMessage.class);
      //perform update initial models
    });
  }

  private <T extends Message> T findMessage(final Collection<Message> messages, final Class<T> cls) {
    return messages.stream()
            .filter(cls::isInstance)
            .map(cls::cast)
            .findFirst()
            .orElseThrow(() -> new RuntimeException(cls.getSimpleName() + " not found"));
  }

  private void skipMessages() {
    while (!endpoint.getMessagesQueue().isEmpty()) {
      final Message message = endpoint.getMessagesQueue().poll();
      LOG.info("skipped message: " + message.type());
    }
  }
}
