package com.spbsu.crawl;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.crawl.bl.GameSession;
import com.spbsu.crawl.bl.Hero;
import com.spbsu.crawl.bl.Mob;
import com.spbsu.crawl.bl.crawlSystemView.SystemView;
import com.spbsu.crawl.data.Command;
import com.spbsu.crawl.data.Message;
import com.spbsu.crawl.data.impl.*;
import com.spbsu.crawl.data.impl.system.*;
import com.spbsu.crawl.learning.LearnDataBuilder;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * User: qdeee
 * Date: 03.04.16
 */
public class GameProcess implements Runnable {
  private static final Logger LOG = Logger.getLogger(GameProcess.class.getName());

  private final WSEndpoint endpoint;
  private final GameSession session;
  private SystemView systemView = new SystemView();
  private LearnDataBuilder learnDataBuilder;

  private int turns;

  public GameProcess(WSEndpoint endpoint, GameSession session) {
    this.endpoint = endpoint;
    this.session = session;
    try {
      learnDataBuilder = new LearnDataBuilder();
    } catch (IOException e) {
      LOG.log(Level.WARNING, "Can't create learn data builder");
      System.exit(1);
    }
  }

  private final FastRandom rng = new FastRandom();

  public void run() {
    skipTo(LobbyComplete.class);
    final String user = rng.nextLowerCaseString(10);
    endpoint.send(new RegisterMessage(user, user, user + "@localhost"));
    skipTo(SetGameLinks.class);
    endpoint.send(new StartGameMessage("dcss-web-trunk"));
    skipTo(PlayerInfoMessage.class);
    endpoint.send(new InputCommandMessage(session.race().select()));
    final Hero.Spec spec = session.spec();
    endpoint.send(new InputCommandMessage(spec.selectProf()));
    if (spec.hasSpec())
      endpoint.send(new InputCommandMessage(spec.selectSpec()));
    Message message;
    systemView.subscribe(session);
    learnDataBuilder.attach(systemView);

    do {
      message = endpoint.poll();
      if (message instanceof Command) {
        ((Command) message).execute(endpoint.out);
      } else if (message instanceof UpdateMapMessage) {
        systemView.updater().message((UpdateMapMessage) message);
      } else if (message instanceof PlayerInfoMessage) {
        systemView.updater().message((PlayerInfoMessage) message);
      } else if (message instanceof MenuMessage) {
        endpoint.send(new KeyMessage(KeyCode.ESCAPE));
      } else if (message instanceof GameEnded) {
        learnDataBuilder.endGame();
        break;
      } else if (message instanceof InputModeMessage) {
        turns++;
        switch (((InputModeMessage) message).inputMode()) {
          case 0:
            break;
          case 1:
            final Mob.Action tickAction = session.action();
            endpoint.send(new InputCommandMessage(tickAction.code()));
            systemView.playerActionView().action(tickAction);
            break;
          case 5:
            endpoint.send(new InputCommandMessage(' '));
            break;
          case 7:
            final Hero.Stat stat = session.chooseStatForUpgrade();
            endpoint.send(new InputCommandMessage(stat.select()));
            break;
          case 8:
            endpoint.send(new InputCommandMessage('Y'));
          default:
            LOG.warning("Unknown input mode received: " + message.json());
        }
      }
    }
    while (true);
  }

  private <T extends Message> T skipTo(Class<T> aClass) {
    Message message;
    do {
      message = endpoint.poll();
      if (message instanceof Command) {
        ((Command) message).execute(endpoint.out);
      } else {
//        LOG.info("Skipped message: " + message.toString());
      }
    }
    while (!aClass.isAssignableFrom(message.getClass()));
    //noinspection unchecked
    return (T) message;
  }

  public double score() {
    return systemView.mapView().knownCells() / (double) turns;
  }
}
